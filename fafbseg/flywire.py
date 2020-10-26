# A collection of tools to interface with manually traced and autosegmented data
# in FAFB.
#
#    Copyright (C) 2019 Philipp Schlegel
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
import pymaid
import navis
import requests

import cloudvolume as cv
import numpy as np
import pandas as pd
import trimesh as tm

from requests_futures.sessions import FuturesSession
from urllib.parse import urlparse, parse_qs
from tqdm.auto import tqdm

from . import xform
from .merge import merge_neuron
from .segmentation import GSPointLoader

try:
    import skeletor as sk
except ImportError:
    sk = None
except BaseException:
    raise


def decode_ngl_url(url, ret='brief'):
    """Decode neuroglancer URL.

    Parameters
    ----------
    url :       str
                URL to decode. Can be shortened URL.
    ret :       "brief" | "full"
                If "brief", will only return "position" (in voxels), "selected"
                segment IDs and "annotations". If full, will return entire scene.

    Returns
    -------
    dict

    """
    assert isinstance(url, (str, dict))
    assert ret in ['brief', 'full']

    query = parse_qs(urlparse(url).query, keep_blank_values=True)

    if 'json_url' in query:
        # Fetch state
        token = cv.secrets.chunkedgraph_credentials['token']
        r = requests.get(query['json_url'][0], headers={'Authorization': f"Bearer {token}"})
        r.raise_for_status()

        scene = r.json()
    else:
        scene = query

    if ret == 'brief':
        seg_layers = [l for l in scene['layers'] if l.get('type') == 'segmentation_with_graph']
        an_layers = [l for l in scene['layers'] if l.get('type') == 'annotation']
        return {'position': scene['navigation']['pose']['position']['voxelCoordinates'],
                'annotations': [a for l in an_layers for a in l.get('annotations', [])],
                'selected': [s for l in seg_layers for s in l.get('segments', [])]}

    return scene


def fetch_edit_history(x, progress=True, max_threads=4):
    """Fetch edit history for given neuron(s).

    Parameters
    ----------
    x :             int | iterable
    progress :      bool
                    If True, show progress bar.
    max_threads :   int
                    Max number of parallel requests to server.

    Returns
    -------
    pandas.DataFrame

    """
    if not isinstance(x, (list, set, np.ndarray)):
        x = [x]

    session = requests.Session()
    future_session = FuturesSession(session=session, max_workers=max_threads)
    token = cv.secrets.chunkedgraph_credentials['token']
    session.headers['Authorization'] = f"Bearer {token}"

    futures = []
    for i in x:
        url = f'https://prodv1.flywire-daf.com/segmentation/api/v1/table/fly_v31/root/{i}/tabular_change_log'
        f = future_session.get(url, params=None)
        futures.append(f)

    # Get the responses
    resp = [f.result() for f in tqdm(futures,
                                     desc='Fetching',
                                     disable=not progress or len(futures) == 1,
                                     leave=False)]

    df = []
    for r, i in zip(resp, x):
        r.raise_for_status()
        this_df = pd.DataFrame(r.json())
        this_df['segment'] = i
        if not this_df.empty:
            df.append(this_df)

    df = pd.concat(df, axis=0, sort=True)
    df['timestamp'] = pd.to_datetime(df.timestamp, unit='ms')

    return df


def locs_to_segments(locs, root_ids=True, vol='graphene://https://prodv1.flywire-daf.com/segmentation/1.0/fly_v31',
                     progress=True, coordinates='pixel', max_workers=8, **kwargs):
    """Retrieve flywire IDs at given location(s).

    Parameters
    ----------
    locs :          list-like
                    Array of x/y/z coordinates.
    root_ids :      bool
                    If True, will return root IDs. If False, will return supervoxel
                    IDs.
    vol :           str | CloudVolume
    progress :      bool
                    If True, shows progress bar.
    coordinates :   "pixel" | "nm"
                    Units in which your coordinates are in. "pixel" is assumed
                    to be 4x4x40 (x/y/z) nanometers.
    max_workers :   int
                    How many parallel requests we can make to the segmentation source.
    **kwargs
                    Keyword arguments are passed on to the cloudvolume.

    Returns
    -------
    list
                List of segmentation IDs in the same order as ``locs``.

    """
    assert coordinates in ['nm', 'pixel']

    locs = np.array(locs)
    assert locs.shape[1] == 3

    # Note: do NOT use the "/table/" version
    vol = _parse_volume(vol, **kwargs)

    # GSPointLoader expects nanometer -> does the mapping based on mip itself
    if coordinates == 'pixel':
        locs = (locs * [4, 4, 40]).astype(int)

    pl = GSPointLoader(fw_vol)
    pl.add_points(locs)

    _, svoxels = pl.load_all(max_workers=max_workers,
                             progress=progress,
                             return_sorted=True)

    svoxels = svoxels.flatten()

    if not root_ids:
        return svoxels

    # get_roots() doesn't like to be asked for zeros - cases server error
    roots = np.zeros(svoxels.shape, dtype=np.int64)
    roots[svoxels != 0] = fw_vol.get_roots(svoxels[svoxels != 0])

    return roots


def skid_to_id(skid,
               vol='graphene://https://prodv1.flywire-daf.com/segmentation/1.0/fly_v31',
               progress=True, **kwargs):
    """Find the flywire ID(s) corresponding to given CATMAID skeleton ID(s).

    This function works by:
        1. Fetch supervoxels for all nodes in the CATMAID skeletons
        2. Pick a random sample of ``sample`` of these supervoxels
        3. Fetch the most recent root IDs for the sample supervoxels
        4. Return the root ID that collectively cover 90% of the supervoxels

    Parameters
    ----------
    id :            int | list-like | str | TreeNeuron/List
                    Anything that's not a TreeNeuron/List be passed directly
                    to ``pymaid.get_neuron``.
    vol :           str | CloudVolume
    progress :      bool
                    If True, shows progress bar.

    Returns
    -------
    pandas.DataFrame
                    Mapping of flywire IDs to skeleton IDs with confidence::

                      flywire_id   skeleton_id   confidence
                    0
                    1

    """
    vol = _parse_volume(vol, **kwargs)

    if not isinstance(skid, (navis.TreeNeuron, navis.NeuronList)):
        skid = pymaid.get_neuron(skid)

    if isinstance(skid, navis.TreeNeuron):
        nodes = skid.nodes[['x', 'y', 'z']]
        nodes['skeleton_id'] = skid.id
    elif isinstance(skid, navis.NeuronList):
        nodes = skid.nodes[['x', 'y', 'z']]
    else:
        raise TypeError(f'Unable to data of type "{type(skid)}"')

    # XForm coordinates from FAFB14 to FAFB14.1
    nodes[['xf', 'yf', 'zf']] = xform.fafb14_to_flywire(nodes[['x', 'y', 'z']].values,
                                                        coordinates='nanometers')

    # Get the root IDs for each of these locations
    nodes['root_id'] = locs_to_segments(nodes[['xf', 'yf', 'zf']].values,
                                        coordinates='nm')

    # Get supervoxel ids - we need to use mip=0 because otherwise small neurons might
    # not have any (visible) supervoxels
    svoxels = vol.get_leaves(id, bbox=vol.meta.bounds(0), mip=0)

    # Shuffle voxels
    np.random.shuffle(svoxels)

    # Generate sample
    if sample >= 1:
        smpl = svoxels[: sample]
    else:
        smpl = svoxels[: int(len(svoxels) * sample)]

    # Fetch up-to-date root IDs for the sampled supervoxels
    roots = vol.get_roots(smpl)

    # Find unique Ids and count them
    unique, counts = np.unique(roots, return_counts=True)

    # Get sorted indices
    sort_ix = np.argsort(counts)

    # New Id is the most frequent ID
    new_id = unique[sort_ix[-1]]

    # Confidence is the difference between the top and the 2bd most frequent ID
    if len(unique) > 1:
        conf = round((counts[sort_ix[-1]] - counts[sort_ix[-2]]) / sum(counts),
                     2)
    else:
        conf = 1

    return pd.DataFrame([[id, new_id, conf, id != new_id]],
                        columns=['old_id', 'new_id', 'confidence', 'changed'])


def update_ids(id,
               sample=0.1,
               vol='graphene://https://prodv1.flywire-daf.com/segmentation/table/fly_v31',
               progress=True, **kwargs):
    """Retrieve the most recent version of given flywire neuron(s).

    This function works by:
        1. Fetching all supervoxels for a given to-be-updated ID
        2. Picking a random sample of ``sample`` of these supervoxels
        3. Fetching the most recent root IDs for the sample supervoxels
        4. Returning the root ID that was hit the most.

    Parameters
    ----------
    id :            int | list-like
                    Single ID or list of flywire (root) IDs.
    sample :        int | float
                    Number (>= 1) or fraction (< 1) of super voxels to sample
                    to guess the most recent version.
    vol :           str | CloudVolume
    progress :      bool
                    If True, shows progress bar.

    Returns
    -------
    pandas.DataFrame
                    Mapping of old -> new root IDs with confidence::

                      old_id   new_id   confidence   changed
                    0
                    1

    """
    assert sample > 0, '`sample` must be > 0'

    vol = _parse_volume(vol, **kwargs)

    if isinstance(id, (list, set, np.ndarray)):
        res = [update_ids(x,
                          vol=vol,
                          sample=sample) for x in tqdm(id,
                                                       desc='Updating',
                                                       leave=False,
                                                       disable=not progress or len(id) == 1)]
        return pd.concat(res, axis=0, sort=False)

    # Get supervoxel ids - we need to use mip=0 because otherwise small neurons might
    # not have any (visible) supervoxels
    svoxels = vol.get_leaves(id, bbox=vol.meta.bounds(0), mip=0)

    # Shuffle voxels
    np.random.shuffle(svoxels)

    # Generate sample
    if sample >= 1:
        smpl = svoxels[: sample]
    else:
        smpl = svoxels[: int(len(svoxels) * sample)]

    # Fetch up-to-date root IDs for the sampled supervoxels
    roots = vol.get_roots(smpl)

    # Find unique Ids and count them
    unique, counts = np.unique(roots, return_counts=True)

    # Get sorted indices
    sort_ix = np.argsort(counts)

    # New Id is the most frequent ID
    new_id = unique[sort_ix[-1]]

    # Confidence is the difference between the top and the 2bd most frequent ID
    if len(unique) > 1:
        conf = round((counts[sort_ix[-1]] - counts[sort_ix[-2]]) / sum(counts),
                     2)
    else:
        conf = 1

    return pd.DataFrame([[id, new_id, conf, id != new_id]],
                        columns=['old_id', 'new_id', 'confidence', 'changed'])


def _merge_flywire_neuron(id, cvpath, target_instance, tag,
                          drop_soma_hairball=True, **kwargs):
    """Merge flywire neuron into FAFB.

    This function (1) fetches a mesh from flywire, (2) turns it into a skeleton,
    (3) maps the coordinates to FAFB 14 and (4) runs ``fafbseg.merge_neuron``
    to merge the skeleton into CATMAID. See Examples below on how to run these
    individual steps yourself if you want more control over e.g. how the mesh
    is skeletonized.

    Parameters
    ----------
    id  :               int
                        ID of the flywire neuron you want to merge.
    cvpath :            str | cloudvolume.CloudVolume
                        Either the path to the flywire segmentation
                        (``graphene://...``) or an already initialized
                        ``CloudVolume``.
    target_instance :    pymaid.CatmaidInstance
                         Instance to merge the neuron into into.
    tag :                str
                         You personal tag to add as annotation once import into
                         CATMAID is complete.
    drop_soma_hairball : bool
                         If True, we will try to drop the hairball that is
                         typically created inside the soma.
    **kwargs
                Keyword arguments are passed on to ``fafbseg.merge_neuron``.

    Examples
    --------
    # Import flywire neuron
    >>> _ = merge_flywire_neuron(id=720575940610453042,
    ...                          cvpath='graphene://https://prodv1.flywire-daf.com/segmentation/1.0/fly_v26',
    ...                          target_instance=manual,
    ...                          tag='WTCam')

    # Run each step yourself
    >>>

    TODO
    ----
    - use mesh to speed up testing overlap and to inform edges in union

    """
    if not sk:
        raise ImportError('Must install skeletor: pip3 install skeletor')

    if isinstance(cvpath, cv.frontends.CloudVolumeGraphene):
        vol = cvpath
    elif isinstance(cvpath, str):
        vol = cv.CloudVolume(cvpath)
    else:
        raise TypeError('Unable to initialize a cloudvolume from "{}"'.format(type(cvpath)))

    # Make sure this is a valid integer
    id = int(id)

    # Download the mesh
    mesh = vol.mesh.get(id, deduplicate_chunk_boundaries=False)[id]

    # Simplify
    simp = sk.simplify(mesh, ratio=.2)

    # Validate before we detect the soma verts
    simp = sk.utilities.fix_mesh(simp, inplace=True)

    # Try detecting the soma
    if drop_soma_hairball:
        soma_verts = _detect_soma(simp)

    # Contract
    cntr = sk.contract(simp,
                       SL=40,
                       WH0=2,
                       epsilon=0.1,
                       precision=1e-7,
                       validate=False)

    # Generate skeleton
    swc = sk.skeletonize(cntr,
                         method='vertex_clusters',
                         sampling_dist=200,
                         vertex_map=True,
                         validate=False)

    # Clean up
    cleaned = sk.clean(swc, mesh=mesh, validate=False)

    # Extract radii
    cleaned['radius'] = sk.radii(cleaned, mesh=mesh, validate=False)

    # Convert to neuron
    n_fw = navis.TreeNeuron(cleaned, id=id, units='nm', soma=None)

    # Drop any nodes that are soma vertices
    if drop_soma_hairball and soma_verts.shape[0] > 0:
        keep = n_fw.nodes.loc[~n_fw.nodes.vertex_id.isin(soma_verts),
                              'node_id'].values
        n_fw = navis.subset_neuron(n_fw, keep)

    # Confirm
    viewer = navis.Viewer(title='Confirm skeletonization')
    # Make sure viewer is actually visible and cleared
    viewer.show()
    viewer.clear()
    # Add original skeleton
    viewer.add(n_fw, color='r')
    viewer.add(navis.MeshNeuron(mesh), color='w', alpha=.2)

    msg = """
    Please carefully inspect the skeletonization of the flywire mesh.
    Hit ENTER to proceed if happy or CTRL-C to cancel.
    """

    try:
        _ = input(msg)
    except KeyboardInterrupt:
        raise KeyboardInterrupt('Merge process aborted by user.')
    except BaseException:
        raise
    finally:
        viewer.close()

    # Xform to FAFB
    n_fafb = xform.flywire_to_fafb14(n_fw, on_fail='raise', coordinates='nm', inplace=False)
    mesh_fafb = xform.flywire_to_fafb14(tm.Trimesh(mesh.vertices, mesh.faces),
                                        on_fail='raise', coordinates='nm', inplace=False)    

    # Heal neuron
    n_fafb = navis.heal_fragmented_neuron(n_fafb)

    # Merge neuron
    return merge_neuron(n_fafb, target_instance=target_instance, tag=tag,
                        mesh=mesh_fafb, **kwargs)


def _detect_soma(mesh):
    """Tries detecting the soma based on vertex clusters.

    Parameters
    ----------
    mesh :      trimesh.Trimesh | navis.MeshNeuron
                Coordinates are assumed to be in nanometers. Mesh must not be
                downsampled.

    Returns
    -------
    vertex indices

    """
    # Build a KD tree
    from scipy.spatial import cKDTree
    tree = cKDTree(mesh.vertices)

    # Find out how many neighbours each vertex has within a 3 micron radius
    n_neighbors = tree.query_ball_point(mesh.vertices,
                                        r=4000,
                                        return_length=True,
                                        n_jobs=3)

    # Seed for soma is the node with the most neighbors
    seed = np.argmax(n_neighbors)

    # We need to find a sensible threshold for neurons without an actual soma
    res = np.mean(mesh.area_faces)
    if n_neighbors.max() < (20e4 / res):
        return np.array([])

    # Find nodes within 10 microns of the seed
    dist, ix = tree.query(mesh.vertices[[seed]],
                          k=mesh.vertices.shape[0],
                          distance_upper_bound=10000)
    soma_verts = ix[dist < float('inf')]

    """
    TODO:
    - use along-the-mesh distances instead to avoid pulling in close-by neurites
    - combine this with looking for a fall-off in N neighbors, i.e. when we
      hit the primary neurite track
    """

    return soma_verts


def _parse_volume(vol, **kwargs):
    """Parse CloudVolume."""
    global fw_vol
    if 'CloudVolume' not in str(type(vol)):
        #  Change default volume if necessary
        if not fw_vol or getattr(fw_vol, 'path') != vol:
            # Set and update defaults from kwargs
            defaults = dict(mip=0,
                            fill_missing=True,
                            use_https=True,  # this way google secret is not needed
                            progress=False)
            defaults.update(kwargs)

            fw_vol = cv.CloudVolume(vol, **defaults)
            fw_vol.path = vol
    else:
        fw_vol = vol
    return fw_vol


# Initialize without a volume
fw_vol = None
