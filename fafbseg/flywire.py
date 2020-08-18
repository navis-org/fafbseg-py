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
import requests

import cloudvolume as cv
import numpy as np
import pandas as pd

from requests_futures.sessions import FuturesSession
from urllib.parse import urlparse, parse_qs
from tqdm.auto import tqdm

from .mapping import xform_flywire_fafb14
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
                If brief, will only return "position" (in voxels), "selected"
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

    Returns
    -------
    list
                List of segmentation IDs in the same order as ``locs``.

    """
    assert coordinates in ['nm', 'pixel']

    locs = np.array(locs)
    assert locs.shape[1] == 3

    global fw_vol
    if 'CloudVolume' not in str(type(vol)):
        #  Change default volume if necessary
        if not fw_vol or getattr(fw_vol, 'path') != vol:
            # Set and update defaults from kwargs
            defaults = dict(cache=True,
                            mip=0,
                            fill_missing=True,
                            use_https=True,  # this way google secret is not needed
                            progress=False)
            defaults.update(kwargs)

            fw_vol = cv.CloudVolume(vol, **defaults)
            fw_vol.path = vol
    else:
        fw_vol = vol

    # GSPointLoader expects nanometer -> does the mapping based on mip itself
    if coordinates == 'pixel':
        locs = (locs * [4, 4, 40]).astype(int)

    pl = GSPointLoader(fw_vol)
    pl.add_points(locs)

    _, svoxels = pl.load_all(max_workers=max_workers,
                             progress=progress,
                             return_sorted=True)

    if not root_ids:
        return svoxels

    # get_roots() doesn't like to be asked for zeros - cases server error
    roots = np.zeros(svoxels.shape)
    roots[svoxels != 0] = fw_vol.get_roots(svoxels[svoxels != 0])

    return roots


def __merge_flywire_neuron(id, cvpath, **kwargs):
    """Merge flywire neuron into FAFB.

    This function (1) fetches a mesh from flywire, (2) turns it into a skeleton,
    (3) maps the coordinates to FAFB 14 and (4) runs ``fafbseg.merge_neuron``
    to merge the skeleton into CATMAID. See Examples below on how to run these
    individual steps yourself if you want more control over e.g. how the mesh
    is skeletonized.

    Parameters
    ----------
    id  :       int
                ID of the neuron you want to merge.
    cvpath :    str | cloudvolume.CloudVolume
                Either the path to the flywire segmentation (``graphene://...``)
                or an already initialized ``CloudVolume``.
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

    """
    if not sk:
        raise ImportError('Must install skeletor: pip3 install skeletor')

    if isinstance(cvpath, cv.CloudVolume):
        vol = cvpath
    elif isinstance(cvpath, str):
        vol = cv.CloudVolume(cvpath)
    else:
        raise TypeError('Unable to initialize a cloudvolume from "{}"'.format(type(cvpath)))

    # Make sure this is a valid integer
    id = int(id)

    # Download the mesh
    mesh = vol.mesh.get(id)[id]

    # Contract
    cntr = sk.contract(mesh)

    # Generate skeleton
    swc = sk.skeletonize(cntr, method='vertex_clustering', sampling_dist=100)

    # Clean up
    cleaned = sk.clean(swc, mesh=mesh)

    # Extract radii
    cleaned['radius'] = sk.radius(cleaned, mesh=mesh)

    # Convert to neuron
    n_fw = pymaid.from_swc(cleaned, neuron_id=id)

    # Xform to FAFB
    n_fafb = xform_flywire_fafb14(n_fw, on_fail='raise', coordinates='nm', inplace=False)

    # Merge neuron
    return merge_neuron(n_fafb, **kwargs)


fw_vol = None
