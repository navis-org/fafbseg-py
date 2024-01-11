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
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU General Public License for more details.

"""Functions to extract skeletons from L2 graphs.

Heavily borrows from code from Casey Schneider-Mizell's "pcg_skel"
(https://github.com/AllenInstitute/pcg_skel).

"""

import navis
import fastremap

import networkx as nx
import numpy as np
import pandas as pd
import skeletor as sk
import trimesh as tm

from concurrent.futures import ThreadPoolExecutor
from functools import partial

from .annotations import parse_neuroncriteria
from .utils import get_cloudvolume, get_cave_client, retry, inject_dataset

__all__ = ['get_l2_skeleton', 'get_l2_dotprops', 'get_l2_graph', 'get_l2_info',
           'find_anchor_loc']


@parse_neuroncriteria()
@inject_dataset()
def get_l2_info(root_ids, progress=True, max_threads=4, *, dataset=None):
    """Fetch basic info for given neuron(s) using the L2 cache.

    Parameters
    ----------
    root_ids  :     int | list of ints | NeuronCriteria
                    FlyWire root ID(s) for which to fetch L2 infos.
    progress :      bool
                    Whether to show a progress bar.
    max_threads :   int
                    Number of parallel requests to make.
    dataset :       "public" | "production" | "sandbox" | "flat_630", optional
                    Against which FlyWire dataset to query. If ``None`` will fall
                    back to the default dataset (see
                    :func:`~fafbseg.flywire.set_default_dataset`).

    Returns
    -------
    pandas.DataFrame
                        DataFrame with basic info (also see Examples):
                          - `length_um` is the sum of the max diameter across
                            all L2 chunks; note that this severely
                            underestimates the actual length (factor >10) but is
                            still useful for relative comparisons
                          - `bounds_nm` is a very rough bounding box based on the
                            representative coordinates of the L2 chunks
                          - `chunks_missing` is the number of L2 chunks not
                            present in the L2 cache

    Examples
    --------
    >>> from fafbseg import flywire
    >>> info = flywire.get_l2_info(720575940614131061)
    >>> info                                                    # doctest: +SKIP
                  root_id  l2_chunks  chunks_missing    area_um2    size_um3  length_um   ...
    0  720575940614131061        286               0  2378.16384  163.876526     60.666   ...

    """
    if navis.utils.is_iterable(root_ids):
        root_ids = np.unique(root_ids)
        info = []
        with ThreadPoolExecutor(max_workers=max_threads) as pool:
            func = partial(get_l2_info, dataset=dataset)
            futures = pool.map(func, root_ids)
            info = [
                f
                for f in navis.config.tqdm(
                    futures,
                    desc="Fetching L2 info",
                    total=len(root_ids),
                    disable=not progress or len(root_ids) == 1,
                    leave=False,
                )
            ]
        return pd.concat(info, axis=0).reset_index(drop=True)

    # Get/Initialize the CAVE client
    client = get_cave_client(dataset=dataset)

    get_l2_ids = partial(retry(client.chunkedgraph.get_leaves), stop_layer=2)
    l2_ids = get_l2_ids(root_ids)

    attributes = ["area_nm2", "size_nm3", "max_dt_nm", "rep_coord_nm"]
    get_l2data = retry(client.l2cache.get_l2data)
    info = get_l2data(l2_ids.tolist(), attributes=attributes)
    n_miss = len([v for v in info.values() if not v])

    row = [root_ids, len(l2_ids), n_miss]
    info_df = pd.DataFrame([row], columns=["root_id", "l2_chunks", "chunks_missing"])

    # Collect L2 attributes
    for at in attributes:
        if at in ("rep_coord_nm",):
            continue

        summed = sum([v.get(at, 0) for v in info.values()])
        if at.endswith("3"):
            summed /= 1000**3
        elif at.endswith("2"):
            summed /= 1000**2
        else:
            summed /= 1000

        info_df[at.replace("_nm", "_um")] = [summed]

    # Check bounding box
    pts = np.array([v["rep_coord_nm"] for v in info.values() if v])

    if len(pts) > 1:
        bounds = [v for l in zip(pts.min(axis=0), pts.max(axis=0)) for v in l]
    elif len(pts) == 1:
        pt = pts[0]
        rad = [v["max_dt_nm"] for v in info.values() if v][0] / 2
        bounds = [
            pt[0] - rad,
            pt[0] + rad,
            pt[1] - rad,
            pt[1] + rad,
            pt[2] - rad,
            pt[2] + rad,
        ]
        bounds = [int(co) for co in bounds]
    else:
        bounds = None
    info_df["bounds_nm"] = [bounds]

    info_df.rename({"max_dt_um": "length_um"}, axis=1, inplace=True)

    return info_df


@parse_neuroncriteria()
@inject_dataset()
def get_l2_chunk_info(l2_ids, progress=True, chunk_size=2000, *, dataset=None):
    """Fetch info for given L2 chunks.

    Parameters
    ----------
    l2_ids  :   int | list of ints | NeuronCriteria
                FlyWire root ID(s) for which to fetch L2 infos.
    progress :  bool
                Whether to show a progress bar.
    chunksize : int
                Number of L2 IDs per query.
    dataset :   "public" | "production" | "sandbox" | "flat_630", optional
                Against which FlyWire dataset to query. If ``None`` will fall
                back to the default dataset (see
                :func:`~fafbseg.flywire.set_default_dataset`).

    Returns
    -------
    pandas.DataFrame

    """
    # Get/Initialize the CAVE client
    client = get_cave_client(dataset=dataset)

    # Get the L2 representative coordinates, vectors and (if required) volume
    attributes = ['rep_coord_nm', 'pca', 'size_nm3']

    l2_info = {}
    with navis.config.tqdm(desc='Fetching L2 info',
                           disable=not progress,
                           total=len(l2_ids),
                           leave=False) as pbar:
        func = retry(client.l2cache.get_l2data)
        for chunk_ix in np.arange(0, len(l2_ids), chunk_size):
            chunk = l2_ids[chunk_ix: chunk_ix + chunk_size]
            l2_info.update(func(chunk.tolist(), attributes=attributes))
            pbar.update(len(chunk))

    # L2 chunks without info will show as empty dictionaries
    # Let's drop them to make our life easier (speeds up indexing too)
    l2_info = {k: v for k, v in l2_info.items() if v}

    if l2_info:
        pts = np.vstack([i['rep_coord_nm'] for i in l2_info.values()])
        vec = np.vstack([i.get('pca', [[None, None, None]])[0] for i in l2_info.values()])
        sizes = np.array([i['size_nm3'] for i in l2_info.values()])

        info_df = pd.DataFrame()
        info_df['id'] = list(l2_info.keys())
        info_df['x'] = (pts[:, 0] / 4).astype(int)
        info_df['y'] = (pts[:, 1] / 4).astype(int)
        info_df['z'] = (pts[:, 2] / 40).astype(int)
        info_df['vec_x'] = vec[:, 0]
        info_df['vec_y'] = vec[:, 1]
        info_df['vec_z'] = vec[:, 2]
        info_df['size_nm3'] = sizes
    else:
        info_df = pd.DataFrame([], columns=['id',
                                            'x', 'y', 'z',
                                            'vec_x', 'vec_y', 'vec_z',
                                            'size_nm3'])

    return info_df


@parse_neuroncriteria()
@inject_dataset()
def find_anchor_loc(root_ids,
                    validate=False,
                    max_threads=4,
                    progress=True,
                    *,
                    dataset=None):
    """Find a representative coordinate.

    This works by querying the L2 cache and using the representative coordinate
    for the largest L2 chunk.

    Parameters
    ----------
    root_ids :      int | list thereof | NeuronCriteria
                    Root ID(s) to get coordinate for.
    validate :      bool
                    If True, will validate the x/y/z position. I have yet to
                    encounter a representative coordinate that wasn't mapping
                    to the correct L2 chunk - therefore this parameter is False
                    by default.
    max_threads :   int
                    Number of parallel threads to use.
    dataset :       "public" | "production" | "sandbox" | "flat_630", optional
                    Against which FlyWire dataset to query. If ``None`` will fall
                    back to the default dataset (see
                    :func:`~fafbseg.flywire.set_default_dataset`).

    Returns
    -------
    pandas.DataFrame

    """
    if navis.utils.is_iterable(root_ids):
        root_ids = np.asarray(root_ids).astype(np.int64)
        root_ids_unique = np.unique(root_ids)
        info = []
        with ThreadPoolExecutor(max_workers=max_threads) as pool:
            func = partial(find_anchor_loc,
                           dataset=dataset,
                           validate=False,
                           progress=False)
            futures = pool.map(func, root_ids_unique)
            info = [f for f in navis.config.tqdm(futures,
                                                 desc='Fetching locations',
                                                 total=len(root_ids_unique),
                                                 disable=not progress or len(root_ids_unique) == 1,
                                                 leave=False)]
        df = pd.concat(info, axis=0, ignore_index=True)

        # Validate
        if validate:
            has_loc = ~df.x.isnull()
            if any(has_loc):
                from .segmentation import locs_to_supervoxels
                sv = locs_to_supervoxels(df.loc[has_loc, ['x', 'y', 'z']].values)
                df['supervoxel'] = None
                df.loc[has_loc, 'supervoxel'] = sv.astype(str)  # do not change str

                # Get/Initialize the CAVE client
                client = get_cave_client(dataset=dataset)

                # Get root timestamps
                ts = client.chunkedgraph.get_root_timestamps(df.root_id.values.tolist())

                df['valid'] = False
                for i in navis.config.trange(len(df),
                                             desc='Validating',
                                             disable=not progress or len(df) == 1,
                                             leave=False):
                    if df.supervoxel.values[i]:
                        sv = np.int64(df.supervoxel.values[i])
                        r = client.chunkedgraph.get_root_id(sv, timestamp=ts[i])
                        df.loc[i, 'valid'] = r == df.root_id.values[i]

        # Make sure the original order is retained
        df = df.set_index('root_id').loc[root_ids].reset_index(drop=False)

        return df

    root_ids = np.int64(root_ids)

    # Get/Initialize the CAVE client
    client = get_cave_client(dataset=dataset)

    get_l2_ids = partial(retry(client.chunkedgraph.get_leaves), stop_layer=2)
    l2_ids = get_l2_ids(root_ids)

    get_l2data = retry(get_l2_chunk_info)
    info = get_l2data(l2_ids, progress=progress)

    if info.empty:
        loc = [None, None, None]
    else:
        info.sort_values('size_nm3', ascending=False, inplace=True)
        loc = info[['x', 'y', 'z']].values[0].tolist()

    df = pd.DataFrame([[root_ids] + loc],
                      columns=['root_id', 'x', 'y', 'z'])

    if validate:
        if not isinstance(loc[0], type(None)):
            from .segmentation import locs_to_supervoxels
            sv = locs_to_supervoxels([loc])[0]
            df['supervoxel'] = sv

            if sv:
                ts = client.chunkedgraph.get_root_timestamps(root_ids)[0]
                r = client.chunkedgraph.get_root_id(sv, timestamp=ts)
                df['valid'] = r == root_ids
            else:
                df['valid'] = False

    return df


@parse_neuroncriteria()
@inject_dataset()
def get_l2_graph(root_ids, progress=True, *, dataset=None):
    """Fetch L2 graph(s).

    Parameters
    ----------
    root_ids  : int | list of ints | NeuronCriteria
                FlyWire root ID(s) for which to fetch the L2 graphs.
    progress :  bool
                Whether to show a progress bar.
    dataset :   "public" | "production" | "sandbox" | "flat_630", optional
                Against which FlyWire dataset to query. If ``None`` will fall
                back to the default dataset (see
                :func:`~fafbseg.flywire.set_default_dataset`).

    Returns
    -------
    networkx.Graph
                        The L2 graph or list thereof.

    Examples
    --------
    >>> from fafbseg import flywire
    >>> G = flywire.get_l2_graph(720575940614131061)

    """
    if navis.utils.is_iterable(root_ids):
        graphs = []
        for id in navis.config.tqdm(root_ids, desc='Fetching',
                                    disable=not progress or len(root_ids) == 1,
                                    leave=False):
            n = get_l2_graph(id, dataset=dataset)
            graphs.append(n)
        return graphs

    # Get/Initialize the CAVE client
    client = get_cave_client(dataset=dataset)

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


@parse_neuroncriteria()
@inject_dataset()
def get_l2_skeleton(root_id, refine=True, drop_missing=True, l2_node_ids=False,
                omit_failures=None, progress=True, max_threads=4,
                *, dataset=None, **kwargs):
    """Generate skeleton from L2 graph.

    Parameters
    ----------
    root_id  :      int | list of ints | NeuronCriteria
                    Root ID(s) of the FlyWire neuron(s) you want to
                    skeletonize.
    refine :        bool
                    If True, will refine skeleton nodes by moving them in
                    the center of their corresponding chunk meshes. This
                    uses the L2 cache (see :func:`fafbseg.flywire.get_l2_info`).
    drop_missing :  bool
                    Only relevant if ``refine=True``: If True, will drop
                    chunks that don't exist in the L2 cache. These are
                    typically chunks that are either very small or new.
                    If False, chunks missing from L2 cache will be kept but
                    with their unrefined, approximate position.
    l2_node_ids :   bool
                    If True, will use the L2 IDs as node IDs (instead of
                    just enumerating the nodes).
    omit_failures : bool, optional
                    Determine behaviour when skeleton generation fails
                    (e.g. if the neuron has only a single chunk):
                        - ``None`` (default) will raise an exception
                        - ``True`` will skip the offending neuron (might result
                        in an empty ``NeuronList``)
                        - ``False`` will return an empty ``TreeNeuron``
    progress :      bool
                    Whether to show a progress bar.
    max_threads :   int
                    Number of parallel requests to make when fetching the
                    L2 skeletons.
    dataset :       "public" | "production" | "sandbox" | "flat_630", optional
                    Against which FlyWire dataset to query. If ``None`` will fall
                    back to the default dataset (see
                    :func:`~fafbseg.flywire.set_default_dataset`).
    **kwargs
                    Keyword arguments are passed through to the `TreeNeuron`
                    initialization. Use to e.g. set extra properties.

    Returns
    -------
    skeleton(s) :       navis.TreeNeuron | navis.NeuronList
                        The extracted L2 skeleton.

    See Also
    --------
    :func:`fafbseg.flywire.get_l2_dotprops`
                        Create dotprops instead of skeletons (faster and
                        possibly more accurate).
    :func:`~fafbseg.flywire.get_skeletons`
                        Fetch precomputed full resolution skeletons. Only
                        available for proofread neurons and for certain
                        materialization versions.
    :func:`fafbseg.flywire.skeletonize_neuron`
                        Skeletonize the full resolution mesh.

    Examples
    --------
    >>> from fafbseg import flywire
    >>> n = flywire.get_l2_skeleton(720575940614131061)

    """
    if omit_failures not in (None, True, False):
        raise ValueError('`omit_failures` must be either None, True or False. '
                         f'Got "{omit_failures}".')

    if navis.utils.is_iterable(root_id):
        root_id = np.asarray(root_id, dtype=np.int64)

        get_l2_skels = partial(get_l2_skeleton, refine=refine, drop_missing=drop_missing,
                               omit_failures=omit_failures, dataset=dataset, **kwargs)
        if (max_threads > 1) and (len(root_id) > 1):
            with ThreadPoolExecutor(max_workers=max_threads) as pool:
                futures = pool.map(get_l2_skels, root_id)
                nl = [f for f in navis.config.tqdm(futures,
                                                   desc='Fetching L2 skeletons',
                                                   total=len(root_id),
                                                   disable=not progress or len(root_id) == 1,
                                                   leave=False)]
        else:
            nl = [get_l2_skels(r) for r in navis.config.tqdm(root_id,
                                               desc='Fetching L2 skeletons',
                                               total=len(root_id),
                                               disable=not progress or len(root_id) == 1,
                                               leave=False)]

        # Turn into neuron list
        nl = navis.NeuronList(nl)

        # Bring in original order
        if len(nl):
            root_id = root_id[np.isin(root_id, nl.id)]
            nl = nl.idx[root_id]

        return nl

    # Turn into integer
    root_id = np.int64(root_id)

    # Get the cloudvolume
    vol = get_cloudvolume(dataset)

    # Get/Initialize the CAVE client
    client = get_cave_client(dataset=dataset)

    # Load the L2 graph for given root ID (this is a (N, 2) array of edges)
    get_l2_edges = retry(client.chunkedgraph.level2_chunk_graph)
    l2_eg = get_l2_edges(root_id)

    # If no edges, we can't create a skeleton
    if not len(l2_eg):
        msg = (f'Unable to create L2 skeleton: root ID {root_id} '
               'consists of only a single L2 chunk.')
        if omit_failures is None:
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
        get_l2data = retry(client.l2cache.get_l2data)
        l2_info = get_l2data(l2_ids.tolist(), attributes=['rep_coord_nm', 'max_dt_nm'])
        # Missing L2 chunks will be {'id': {}}
        new_co = {l2dict[np.int64(k)]: v['rep_coord_nm'] for k, v in l2_info.items() if v}
        new_r = {l2dict[np.int64(k)]: v.get('max_dt_nm', 0) for k, v in l2_info.items() if v}

        # Map refined coordinates onto the SWC
        has_new = swc.node_id.isin(new_co)

        # Only apply if we actually have new coordinates - otherwise there
        # the datatype is changed to object for some reason...
        if any(has_new):
            swc.loc[has_new, 'x'] = swc.loc[has_new, 'node_id'].map(lambda x: new_co[x][0])
            swc.loc[has_new, 'y'] = swc.loc[has_new, 'node_id'].map(lambda x: new_co[x][1])
            swc.loc[has_new, 'z'] = swc.loc[has_new, 'node_id'].map(lambda x: new_co[x][2])

        swc['radius'] = swc.node_id.map(new_r)

        # Turn into a proper neuron
        tn = navis.TreeNeuron(swc, id=root_id, units='1 nm', **kwargs)

        # Drop nodes that are still at their unrefined chunk position
        if drop_missing:
            frac_refined = has_new.sum() / len(has_new)
            if not any(has_new):
                msg = (f'Unable to refine: no L2 info for root ID {root_id} '
                       'available. Set `drop_missing=False` to use unrefined '
                       'positions.')
                if omit_failures is None:
                    raise ValueError(msg)
                elif omit_failures:
                    return navis.NeuronList([])
                    # If no omission, return empty TreeNeuron
                else:
                    return navis.TreeNeuron(None, id=root_id, units='1 nm', **kwargs)
            elif frac_refined < .5:
                msg = (f'Root ID {root_id} has only {frac_refined:.1%} of their '
                       'L2 IDs in the cache. Set `drop_missing=False` to use '
                       'unrefined positions.')
                navis.config.logger.warning(msg)

            tn = navis.remove_nodes(tn, swc.loc[~has_new, 'node_id'].values)
            tn._l2_chunks_missing = (~has_new).sum()
    else:
        tn = navis.TreeNeuron(swc, id=root_id, units='1 nm', **kwargs)

    if l2_node_ids:
        ixdict = {ii: l2 for ii, l2 in enumerate(l2_ids)}
        tn.nodes['node_id'] = tn.nodes.node_id.map(ixdict)
        tn.nodes['parent_id'] = tn.nodes.parent_id.map(lambda x: ixdict.get(x, -1))

    return tn


@parse_neuroncriteria()
@inject_dataset()
def get_l2_dotprops(root_ids, min_size=None, sample=False, omit_failures=None,
                progress=True, max_threads=10, *, dataset=None, **kwargs):
    """Generate dotprops from L2 chunks.

    L2 chunks not present in the L2 cache or without a `pca` attribute
    (happens for very small chunks) are silently ignored.

    Parameters
    ----------
    root_ids  :     int | list of ints | NeuronCriteria
                    Root ID(s) of the FlyWire neuron(s) you want to
                    dotprops for.
    min_size :      int, optional
                    Minimum size (in nm^3) for the L2 chunks. Smaller chunks
                    will be ignored. This is useful to de-emphasise the
                    finer terminal neurites which typically break into more,
                    smaller chunks and are hence overrepresented. A good
                    value appears to be around 1_000_000.
    sample :        float [0 > 1], optional
                    If float, will create Dotprops based on a fractional
                    sample of the L2 chunks. The sampling is random but
                    deterministic.
    omit_failures : bool, optional
                    Determine behaviour when dotprops generation fails
                    (i.e. if the neuron has no L2 info):
                        - ``None`` (default) will raise an exception
                        - ``True`` will skip the offending neuron (might result
                        in an empty ``NeuronList``)
                        - ``False`` will return an empty ``Dotprops``
    progress :      bool
                    Whether to show a progress bar.
    max_threads :   int
                    Number of parallel requests to make when fetching the
                    L2 IDs (but not the L2 info).
    dataset :       "public" | "production" | "sandbox" | "flat_630", optional
                    Against which FlyWire dataset to query. If ``None`` will fall
                    back to the default dataset (see
                    :func:`~fafbseg.flywire.set_default_dataset`).
    **kwargs
                    Keyword arguments are passed through to the `Dotprops`
                    initialization. Use to e.g. set extra properties.

    Returns
    -------
    dps :           navis.NeuronList
                    List of Dotprops.

    See Also
    --------
    :func:`fafbseg.flywire.get_l2_skeleton`
                    Fetch skeletons instead of dotprops using the L2
                    edges to infer connectivity.
    :func:`fafbseg.flywire.skeletonize_neuron`
                    Skeletonize the full resolution mesh.

    Examples
    --------
    >>> from fafbseg import flywire
    >>> n = flywire.get_l2_dotprops(720575940614131061)

    """
    if omit_failures not in (None, True, False):
        raise ValueError('`omit_failures` must be either None, True or False. '
                         f'Got "{omit_failures}".')

    if not navis.utils.is_iterable(root_ids):
        root_ids = [root_ids]

    root_ids = np.asarray(root_ids, dtype=np.int64)

    if '0' in root_ids or 0 in root_ids:
        raise ValueError('Unable to produce dotprops for root ID 0.')

    # Get/Initialize the CAVE client
    client = get_cave_client(dataset=dataset)

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

    if sample:
        if sample <= 0 or sample >= 1:
            raise ValueError(f'`sample` must be between 0 and 1, got {sample}')

        for i in range(len(l2_ids)):
            # Make the sampling deterministic
            np.random.seed(1985)
            l2_ids[i] = np.random.choice(l2_ids[i],
                                         size=max(1, int(len(l2_ids[i]) * sample)),
                                         replace=False)

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
        get_l2data = retry(client.l2cache.get_l2data)
        for chunk_ix in np.arange(0, len(l2_ids_all), chunk_size):
            chunk = l2_ids_all[chunk_ix: chunk_ix + chunk_size]
            l2_info.update(get_l2data(chunk.tolist(), attributes=attributes))
            pbar.update(len(chunk))

    # L2 chunks without info will show as empty dictionaries
    # Let's drop them to make our life easier (speeds up indexing too)
    # Note that small L2 chunks won't have a `pca` entry
    l2_info = {k: v for k, v in l2_info.items() if 'pca' in v}

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
            if omit_failures is None:
                raise ValueError(msg)

            if not omit_failures:
                # If no omission, add empty Dotprops
                dps.append(navis.Dotprops(None, k=None, id=root,
                                          units='1 nm', **kwargs))
                dps[-1]._l2_chunks_missing = len(ids)
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
        dps[-1]._l2_chunks_missing = len(ids) - len(this_info)

    return navis.NeuronList(dps)


@inject_dataset()
def get_l2_meshes(x, threads=10, progress=True, *, dataset=None):
    """Fetch L2 meshes for a given neuron.

    Parameters
    ----------
    x :         int | str
                Root ID.
    threads :   int
    progress :  bool
    dataset :   "public" | "production" | "sandbox" | "flat_630", optional
                Against which FlyWire dataset to query. If ``None`` will fall
                back to the default dataset (see
                :func:`~fafbseg.flywire.set_default_dataset`).

    Returns
    -------
    navis.NeuronList

    """
    try:
        x = np.int64(x)
    except ValueError:
        raise ValueError(f'Unable to convert root ID {x} to integer')

    # Get/Initialize the CAVE client
    client = get_cave_client(dataset=dataset)

    # Get the cloudvolume
    vol = get_cloudvolume(dataset)

    # Load the L2 IDs
    l2_ids = client.chunkedgraph.get_leaves(x, stop_layer=2)

    with ThreadPoolExecutor(max_workers=threads) as pool:
        mesh_get = retry(vol.mesh.get)
        futures = [pool.submit(mesh_get, i,
                               allow_missing=True,
                               deduplicate_chunk_boundaries=False) for i in l2_ids]

        res = [f.result() for f in navis.config.tqdm(futures,
                                                     disable=not progress,
                                                     leave=False,
                                                     desc='Loading meshes')]

    # Unpack results
    meshes = {k: v for d in res for k, v in d.items()}

    return navis.NeuronList([navis.MeshNeuron(v, id=k) for k, v in meshes.items()])


def _get_l2_centroids(l2_ids, vol, threads=10, progress=True):
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
