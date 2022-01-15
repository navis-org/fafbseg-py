#    A collection of tools to interface with manually traced and autosegmented
#    data in FAFB.
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

"""Functions to extract skeletons from L2 graphs.

Heavily borrows from code from Casey Schneider-Mizell's "pcg_skel"
(https://github.com/AllenInstitute/pcg_skel).

"""

import navis
import fastremap
import time

import networkx as nx
import numpy as np
import pandas as pd
import skeletor as sk
import trimesh as tm

from concurrent.futures import ThreadPoolExecutor
from functools import partial

from .utils import parse_volume, get_cave_client, retry

__all__ = ['l2_skeleton', 'l2_dotprops', 'l2_graph', 'l2_info']


def l2_info(root_ids, progress=True, max_threads=4, dataset='production'):
    """Fetch basic info for given neuron(s) using the L2 cache.

    Parameters
    ----------
    root_ids  :         int | list of ints
                        FlyWire root ID(s) for which to fetch L2 infos.
    progress :          bool
                        Whether to show a progress bar.
    max_threads :       int
                        Number of parallel requests to make.

    Returns
    -------
    pandas.DataFrame
                        DataFrame with basic info (also see Examples):
                          - `length_um` is the sum of the max diameter across
                            all L2 chunks
                          - `bounds_nm` is a very rough bounding box based on the
                            representative coordinates of the L2 chunks
                          - `chunks_missing` is the number of L2 chunks not
                            present in the L2 cache

    Examples
    --------
    >>> from fafbseg import flywire
    >>> info = flywire.l2_info(720575940614131061)
    >>> info
                  root_id  l2_chunks  chunks_missing    area_um2    size_um3  length_um   bounds_nm
    0  720575940614131061        286               2  2364.39616  132.467837     60.271   [305456, 311184, ...

    """
    if navis.utils.is_iterable(root_ids):
        root_ids = np.unique(root_ids)
        info = []
        with ThreadPoolExecutor(max_workers=max_threads) as pool:
            func = retry(partial(l2_info, dataset=dataset))
            futures = pool.map(func, root_ids)
            info = [f for f in navis.config.tqdm(futures,
                                                 desc='Fetching L2 info',
                                                 total=len(root_ids),
                                                 disable=not progress or len(root_ids) == 1,
                                                 leave=False)]
        return pd.concat(info, axis=0)

    # Get/Initialize the CAVE client
    client = get_cave_client(dataset)

    get_l2_ids = partial(retry(client.chunkedgraph.get_leaves), stop_layer=2)
    l2_ids = get_l2_ids(root_ids)

    attributes = ['area_nm2', 'size_nm3', 'max_dt_nm', 'rep_coord_nm']
    info = client.l2cache.get_l2data(l2_ids.tolist(), attributes=attributes)
    n_miss = len([v for v in info.values() if not v])

    row = [root_ids, len(l2_ids), n_miss]
    info_df = pd.DataFrame([row],
                           columns=['root_id', 'l2_chunks', 'chunks_missing'])

    # Collect L2 attributes
    for at in attributes:
        if at in ('rep_coord_nm'):
            continue

        summed = sum([v.get(at, 0) for v in info.values()])
        if at.endswith('3'):
            summed /= 1000**3
        elif at.endswith('2'):
            summed /= 1000**2
        else:
            summed /= 1000

        info_df[at.replace('_nm', '_um')] = [summed]

    # Check bounding box
    pts = np.array([v['rep_coord_nm'] for v in info.values() if v])

    if len(pts) > 1:
        bounds = [v for l in zip(pts.min(axis=0), pts.max(axis=0)) for v in l]
    elif len(pts) == 1:
        pt = pts[0]
        rad = [v['max_dt_nm'] for v in info.values() if v][0] / 2
        bounds = [pt[0] - rad, pt[0] + rad,
                  pt[1] - rad, pt[1] + rad,
                  pt[2] - rad, pt[2] + rad]
        bounds = [int(co) for co in bounds]
    else:
        bounds = None
    info_df['bounds_nm'] = [bounds]

    info_df.rename({'max_dt_um': 'length_um'},
                   axis=1, inplace=True)

    return info_df


def l2_graph(root_ids, progress=True, dataset='production'):
    """Fetch L2 graph(s).

    Parameters
    ----------
    root_ids  :         int | list of ints
                        FlyWire root ID(s) for which to fetch the L2 graphs.
    progress :          bool
                        Whether to show a progress bar.

    Returns
    -------
    networkx.Graph
                        The L2 graph.

    Examples
    --------
    >>> from fafbseg import flywire
    >>> n = flywire.l2_graph(720575940614131061)

    """
    if navis.utils.is_iterable(root_ids):
        graphs = []
        for id in navis.config.tqdm(root_ids, desc='Fetching',
                                    disable=not progress or len(root_ids) == 1,
                                    leave=False):
            n = l2_graph(id, dataset=dataset)
            graphs.append(n)
        return graphs

    # Get/Initialize the CAVE client
    client = get_cave_client(dataset)

    # Load the L2 graph for given root ID
    # This is a (N,2) array of edges
    l2_eg = np.array(client.chunkedgraph.level2_chunk_graph(root_ids))

    # Generate graph
    G = nx.Graph()

    if not len(l2_eg):
        # If no edges, this neuron consists of a single chunk
        # Get the single chunk's ID
        chunks = client.chunkedgraph.get_leaves(root_ids, stop_layer=2)
        G.add_nodes_from(chunks)
    else:
        # Drop duplicate edges
        l2_eg = np.unique(np.sort(l2_eg, axis=1), axis=0)

        G.add_edges_from(l2_eg)

    return G


def l2_skeleton(root_id, refine=True, drop_missing=True, omit_failures=None,
                progress=True, dataset='production', **kwargs):
    """Generate skeleton from L2 graph.

    Parameters
    ----------
    root_id  :          int | list of ints
                        Root ID(s) of the FlyWire neuron(s) you want to
                        skeletonize.
    refine :            bool
                        If True, will refine skeleton nodes by moving them in
                        the center of their corresponding chunk meshes. This
                        uses the L2 cache (see :func:`fafbseg.flywire.l2_info`).
    drop_missing :      bool
                        Only relevant if ``refine=True``: If True, will drop
                        chunks that don't exist in the L2 cache. These are
                        typically chunks that are either very small or new.
                        If False, chunks missing from L2 cache will be kept but
                        with their unrefined, approximate position.
    omit_failures :     bool, optional
                        Determine behaviour when skeleton generation fails
                        (e.g. if the neuron has only a single chunk):
                         - ``None`` (default) will raise an exception
                         - ``True`` will skip the offending neuron (might result
                           in an empty ``NeuronList``)
                         - ``False`` will return an empty ``TreeNeuron``
    progress :          bool
                        Whether to show a progress bar.
    **kwargs
                        Keyword arguments are passed through to Dotprops
                        initialization. Use to e.g. set extra properties.

    Returns
    -------
    skeleton(s) :       navis.TreeNeuron | navis.NeuronList
                        The extracted L2 skeleton.

    See Also
    --------
    :func:`fafbseg.flywire.l2_dotprops`
                        Create dotprops instead of skeletons (faster and
                        possibly more accurate).
    :func:`fafbseg.flywire.skeletonize_neuron`
                        Skeletonize the full resolution mesh.

    Examples
    --------
    >>> from fafbseg import flywire
    >>> n = flywire.l2_skeleton(720575940614131061)

    """
    if omit_failures not in (None, True, False):
        raise ValueError('`omit_failures` must be either None, True or False. '
                         f'Got "{omit_failures}".')
    # TODO:
    # - drop duplicate nodes in unrefined skeleton
    # - use L2 graph to find soma: highest degree is typically the soma

    if navis.utils.is_iterable(root_id):
        nl = []
        for id in navis.config.tqdm(root_id, desc='L2 skeletons',
                                    disable=not progress, leave=False):
            n = l2_skeleton(id, refine=refine, drop_missing=drop_missing,
                            omit_failures=omit_failures,
                            progress=progress, dataset=dataset, **kwargs)
            nl.append(n)
        return navis.NeuronList(nl)

    # Turn into integer
    root_id = int(root_id)

    # Get the cloudvolume
    vol = parse_volume(dataset)

    # Get/Initialize the CAVE client
    client = get_cave_client(dataset)

    # Load the L2 graph for given root ID (this is a (N, 2) array of edges)
    get_l2_edges = retry(client.chunkedgraph.level2_chunk_graph)
    l2_eg = get_l2_edges(root_id)

    # If no edges, we can't create a skeleton
    if not len(l2_eg):
        msg = (f'Unable to create L2 skeleton: root ID {root_id} '
               'consists of only a single L2 chunk.')
        if omit_failures == None:
            raise ValueError(msg)

        navis.config.logger.warning(msg)

        if omit_failures:
            # If omission simply return an empty NeuronList
            return navis.NeuronList([])
            # If no omission, return empty TreeNeuron
        else:
            return navis.TreeNeuron(None, id=root_id, units='1 nm', **kwargs)

    # Drop duplicate edges
    l2_eg = np.unique(np.sort(l2_eg, axis=1), axis=0)

    # Unique L2 IDs
    l2_ids = np.unique(l2_eg)

    # ID to index
    l2dict = {l2: ii for ii, l2 in enumerate(l2_ids)}

    # Remap edge graph to indices
    eg_arr_rm = fastremap.remap(l2_eg, l2dict)

    coords = [np.array(vol.mesh.meta.meta.decode_chunk_position(l)) for l in l2_ids]
    coords = np.vstack(coords)

    # This turns the graph into a hierarchal tree by removing cycles and
    # ensuring all edges point towards a root
    if sk.__version_vector__[0] < 1:
        G = sk.skeletonizers.edges_to_graph(eg_arr_rm)
        swc = sk.skeletonizers.make_swc(G, coords=coords)
    else:
        G = sk.skeletonize.utils.edges_to_graph(eg_arr_rm)
        swc = sk.skeletonize.utils.make_swc(G, coords=coords, reindex=False)

    # Set radius to 0
    swc['radius'] = 0

    # Convert to Euclidian space
    # Dimension of a single chunk
    ch_dims = chunks_to_nm([1, 1, 1], vol) - chunks_to_nm([0, 0, 0], vol)
    ch_dims = np.squeeze(ch_dims)

    xyz = swc[['x', 'y', 'z']].values
    swc[['x', 'y', 'z']] = chunks_to_nm(xyz, vol) + ch_dims / 2

    if refine:
        # Get the L2 representative coordinates
        l2_info = client.l2cache.get_l2data(l2_ids.tolist(), attributes=['rep_coord_nm'])
        # Missing L2 chunks will be {'id': {}}
        new_co = {l2dict[int(k)]: v['rep_coord_nm'] for k, v in l2_info.items() if v}

        # Map refined coordinates onto the SWC
        has_new = swc.node_id.isin(new_co)

        # Only apply if we actually have new coordinates - otherwise there
        # the datatype is changed to object for some reason...
        if any(has_new):
            swc.loc[has_new, 'x'] = swc.loc[has_new, 'node_id'].map(lambda x: new_co[x][0])
            swc.loc[has_new, 'y'] = swc.loc[has_new, 'node_id'].map(lambda x: new_co[x][1])
            swc.loc[has_new, 'z'] = swc.loc[has_new, 'node_id'].map(lambda x: new_co[x][2])

        # Turn into a proper neuron
        tn = navis.TreeNeuron(swc, id=root_id, units='1 nm', **kwargs)

        # Drop nodes that are still at their unrefined chunk position
        if drop_missing:
            if not any(has_new):
                msg = (f'Unable to refine: no L2 info for root ID {root_id} '
                       'available. Set `drop_missing=False` to use unrefined '
                       'positions.')
                if omit_failures == None:
                    raise ValueError(msg)
                elif omit_failures:
                    return navis.NeuronList([])
                    # If no omission, return empty TreeNeuron
                else:
                    return navis.TreeNeuron(None, id=root_id, units='1 nm', **kwargs)

            tn = navis.remove_nodes(tn, swc.loc[~has_new, 'node_id'].values)
    else:
        tn = navis.TreeNeuron(swc, id=root_id, units='1 nm', **kwargs)

    return tn


def l2_dotprops(root_ids, min_size=None, omit_failures=None, progress=True,
                max_threads=10, dataset='production', **kwargs):
    """Generate dotprops from L2 chunks.

    L2 chunks not present in the L2 cache are silently ignored.

    Parameters
    ----------
    root_ids  :         int | list of ints
                        Root ID(s) of the FlyWire neuron(s) you want to
                        dotprops for.
    min_size :          int, optional
                        Minimum size (in nm^3) for the L2 chunks. Smaller chunks
                        will be ignored. This is useful to de-emphasise the
                        finer terminal neurites which typically break into more,
                        smaller chunks and are hence overrepresented. A good
                        value appears to be around 1_000_000.
    omit_failures :     bool, optional
                        Determine behaviour when dotprops generation fails
                        (i.e. if the neuron has no L2 info):
                         - ``None`` (default) will raise an exception
                         - ``True`` will skip the offending neuron (might result
                           in an empty ``NeuronList``)
                         - ``False`` will return an empty ``Dotprops``
    progress :          bool
                        Whether to show a progress bar.
    max_threads :       int
                        Number of parallel requests to make when fetching the
                        L2 IDs (but not the L2 info).
    **kwargs
                        Keyword arguments are passed through to Dotprops
                        initialization. Use to e.g. set extra properties.

    Returns
    -------
    dps :               navis.NeuronList
                        List of Dotprops.

    See Also
    --------
    :func:`fafbseg.flywire.l2_skeleton`
                        Create skeletons instead of dotprops using the L2
                        edges to infer connectivity.
    :func:`fafbseg.flywire.skeletonize_neuron`
                        Skeletonize the full resolution mesh.

    Examples
    --------
    >>> from fafbseg import flywire
    >>> n = flywire.l2_dotprops(720575940614131061)

    """
    if omit_failures not in (None, True, False):
        raise ValueError('`omit_failures` must be either None, True or False. '
                         f'Got "{omit_failures}".')

    if not navis.utils.is_iterable(root_ids):
        root_ids = [root_ids]

    root_ids = np.asarray(root_ids)

    if '0' in root_ids or 0 in root_ids:
        raise ValueError('Unable to produce dotprops for root ID 0.')

    # Get/Initialize the CAVE client
    client = get_cave_client(dataset)

    # Load the L2 IDs
    with ThreadPoolExecutor(max_workers=max_threads) as pool:
        get_l2_ids = partial(retry(client.chunkedgraph.get_leaves), stop_layer=2)
        futures = pool.map(get_l2_ids, root_ids)
        l2_ids = [f for f in navis.config.tqdm(futures,
                                               desc='Fetching L2 IDs',
                                               total=len(root_ids),
                                               disable=not progress or len(root_ids) == 1,
                                               leave=False)]

    # Turn IDs into strings
    l2_ids = [i.astype(str) for i in l2_ids]

    # Flatten into a list of all L2 IDs
    l2_ids_all = np.unique([i for l in l2_ids for i in l])

    # Get the L2 representative coordinates, vectors and (if required) volume
    chunk_size = 2000  # no. of L2 IDs per query (doesn't seem have big impact)
    attributes = ['rep_coord_nm', 'pca']
    if min_size:
        attributes.append('size_nm3')

    l2_info = {}
    with navis.config.tqdm(desc='Fetching L2 vectors',
                           disable=not progress,
                           total=len(l2_ids_all),
                           leave=False) as pbar:
        func = retry(client.l2cache.get_l2data)
        for chunk_ix in np.arange(0, len(l2_ids_all), chunk_size):
            chunk = l2_ids_all[chunk_ix: chunk_ix + chunk_size]
            l2_info.update(func(chunk.tolist(), attributes=attributes))
            pbar.update(len(chunk))

    # L2 chunks without info will show as empty dictionaries
    # Let's drop them to make our life easier (speeds up indexing too)
    l2_info = {k: v for k, v in l2_info.items() if v}

    # Generate dotprops
    dps = []
    for root, ids in navis.config.tqdm(zip(root_ids, l2_ids),
                                       desc='Creating dotprops',
                                       total=len(root_ids),
                                       disable=not progress or len(root_ids) <= 1,
                                       leave=False):
        # Get xyz points and the first component of the PCA as vector
        # Note that first subsetting IDs to what's actually available in
        # `l2_info` is actually slower than doing it like this
        this_info = [l2_info[i] for i in ids if i in l2_info]

        if not len(this_info):
            msg = ('Unable to create L2 dotprops: none of the L2 chunks for '
                   f'root ID {root} are present in the L2 cache.')
            if omit_failures == None:
                raise ValueError(msg)

            if not omit_failures:
                # If no omission, add empty Dotprops
                dps.append(navis.Dotprops(None, k=None, id=root,
                                          units='1 nm', **kwargs))
            continue

        pts = np.vstack([i['rep_coord_nm'] for i in this_info])
        vec = np.vstack([i['pca'][0] for i in this_info])

        # Apply min size filter if requested
        if min_size:
            sizes = np.array([i['size_nm3'] for i in this_info])
            pts = pts[sizes >= min_size]
            vec = vec[sizes >= min_size]

        # Generate the actual dotprops
        dps.append(navis.Dotprops(points=pts, vect=vec, id=root, k=None,
                                  units='1 nm', **kwargs))

    return navis.NeuronList(dps)


def l2_meshes(x, threads=10, dataset='production', progress=True):
    """Fetch L2 meshes for a given neuron.

    Parameters
    ----------
    x :         int | str
                Root ID.
    threads :   int
    progress :  bool

    Returns
    -------
    navis.NeuronList

    """
    try:
        x = int(x)
    except:
        raise ValueError(f'Unable to convert root ID {x} to integer')

    # Get/Initialize the CAVE client
    client = get_cave_client(dataset)

    # Get the cloudvolume
    vol = parse_volume(dataset)

    # Load the L2 IDs
    l2_ids = client.chunkedgraph.get_leaves(x, stop_layer=2)

    with ThreadPoolExecutor(max_workers=threads) as pool:
        futures = [pool.submit(vol.mesh.get, i,
                               allow_missing=True,
                               deduplicate_chunk_boundaries=False) for i in l2_ids]

        res = [f.result() for f in navis.config.tqdm(futures,
                                                     disable=not progress,
                                                     leave=False,
                                                     desc='Loading meshes')]

    # Unpack results
    meshes = {k: v for d in res for k, v in d.items()}

    return navis.NeuronList([navis.MeshNeuron(v, id=k) for k, v in meshes.items()])


def get_L2_centroids(l2_ids, vol, threads=10, progress=True):
    """Fetch L2 meshes and compute centroid."""
    with ThreadPoolExecutor(max_workers=threads) as pool:
        futures = [pool.submit(vol.mesh.get, i,
                               allow_missing=True,
                               deduplicate_chunk_boundaries=False) for i in l2_ids]

        res = [f.result() for f in navis.config.tqdm(futures,
                                                     disable=not progress,
                                                     leave=False,
                                                     desc='Loading meshes')]

    # Unpack results
    meshes = {k: v for d in res for k, v in d.items()}

    # For each mesh find the center of mass and move the corresponding point
    centroids = {}
    for k, m in meshes.items():
        m = tm.Trimesh(m.vertices, m.faces)
        # Do NOT use center_mass here -> garbage if not non-watertight
        centroids[k] = m.centroid

    return centroids


def l2_soma(x, dataset='production', progress=True):
    """DOES NOT WORK. Use the L2 graph to guess the soma location.

    In a nutshell: we use the connectedness (i.e. the degree) of L2 chunks to
    guess a neurons soma. The idea was that L2 chunk with the highest degree is
    in the center of the soma. Unfortunately, that only seems to work if the
    soma is really really big.

    Parameters
    ----------
    x  :                int | nx.Graph | lists thereof
                        The neuron(s) for which to find somas. Can be either:
                            - root ID (s)
                            - networkx L2 graph(s)

    Returns
    -------
    [x, y, z]

    Examples
    --------
    >>> n = flywire.l2_soma(720575940614131061)

    """
    if navis.utils.is_iterable(x):
        res = []
        for id in navis.config.tqdm(x, desc='Finding somas',
                                    disable=not progress, leave=False):
            res.append(l2_soma(id, progress=False))
        return res

    # Get the cloudvolume
    vol = parse_volume(dataset)

    # Get/Initialize the CAVE client
    client = get_cave_client(dataset)

    if not isinstance(x, nx.Graph):
        # Load the L2 graph for given root ID
        # This is a (N,2) array of edges
        l2_eg = np.array(client.chunkedgraph.level2_chunk_graph(x))

        # Drop duplicate edges
        l2_eg = np.unique(np.sort(l2_eg, axis=1), axis=0)

        G = nx.Graph()
        G.add_edges_from(l2_eg)
    else:
        G = x

    # Get degrees from the graph
    degrees = np.array(list(G.degree))
    l2_ids = degrees[:, 0]
    l2_degrees = degrees[:, 1]

    # Find the node with highest degree
    mx_deg = l2_ids[np.argmax(l2_degrees)]

    # Get it's centroid
    centroid = get_L2_centroids([mx_deg], vol, threads=1, progress=False)

    return centroid.get(mx_deg, 'NA')


def chunks_to_nm(xyz_ch, vol, voxel_resolution=[4, 4, 40]):
    """Map a chunk location to Euclidean space.

    Parameters
    ----------
    xyz_ch :            array-like
                        (N, 3) array of chunk indices.
    vol :               cloudvolume.CloudVolume
                        CloudVolume object associated with the chunked space.
    voxel_resolution :  list, optional
                        Voxel resolution.

    Returns
    -------
    np.array
                        (N, 3) array of spatial points.

    """
    mip_scaling = vol.mip_resolution(0) // np.array(voxel_resolution, dtype=int)

    x_vox = np.atleast_2d(xyz_ch) * vol.mesh.meta.meta.graph_chunk_size
    return (
        (x_vox + np.array(vol.mesh.meta.meta.voxel_offset(0)))
        * voxel_resolution
        * mip_scaling
    )
