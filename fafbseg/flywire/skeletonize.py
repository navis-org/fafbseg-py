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

import navis
import numbers
import os
import requests
import inspect
import pathlib

import cloudvolume as cv
import multiprocessing as mp
import networkx as nx
import pandas as pd
import numpy as np
import skeletor as sk
import trimesh as tm

from functools import partial
from concurrent.futures import ThreadPoolExecutor

from .segmentation import snap_to_id, is_latest_root
from .utils import get_cloudvolume, silence_find_mat_version, inject_dataset
from .annotations import get_somas, parse_neuroncriteria

SKELETON_BASE_URL = {'630': "https://flyem.mrc-lmb.cam.ac.uk/flyconnectome/flywire_skeletons_630",
                     '783': "https://flyem.mrc-lmb.cam.ac.uk/flyconnectome/flywire_skeletons_783",}
SKELETON_INFO = {"@type": "neuroglancer_skeletons", "transform": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0], "vertex_attributes": [{"id": "radius", "data_type": "float32", "num_components": 1}]}


__all__ = ['skeletonize_neuron', 'skeletonize_neuron_parallel', 'get_skeletons']


@parse_neuroncriteria()
@inject_dataset()
def skeletonize_neuron(x, shave_skeleton=True, remove_soma_hairball=False,
                       assert_id_match=False, threads=2, save_to=None,
                       progress=True, *, dataset=None, **kwargs):
    """Skeletonize FlyWire neuron.

    Note that this is optimized to be primarily fast which comes at the cost
    of (some) quality. Also note that soma detection is using the nucleus
    segmentation and falls back to a radius-based heuristic if no nucleus is
    found.

    Parameters
    ----------
    x  :                 int | trimesh.TriMesh | list thereof | NeuronCriteria
                         ID(s) or trimesh of the FlyWire neuron(s) you want to
                         skeletonize.
    shave_skeleton :     bool
                         If True, we will attempt to remove any "bristles" on
                         the on the backbone which typically occur if the
                         neurites are very big (or badly segmented).
    remove_soma_hairball : bool
                         If True, we will try to drop the hairball that is
                         typically created inside the soma. Note that while this
                         should work just fine for 99% of neurons, it's not very
                         smart and there is always a small chance that we
                         remove stuff that should not have been removed. Also
                         only works if the neuron has its nucleus annotated
                         (see :func:`fafbseg.flywire.get_somas`).
    assert_id_match :    bool
                         If True, will check if skeleton nodes map to the
                         correct segment ID and if not will move them back into
                         the segment. This is potentially very slow!
    threads :            int
                         Number of parallel threads to use for downloading the
                         meshes.
    save_to :            str, optional
                         If provided will save skeleton as SWC at `{save_to}/{id}.swc`.
    progress :           bool
                         Whether to show a progress bar or not.
    dataset :            str | CloudVolume
                         Against which FlyWire dataset to query::
                           - "production" (current production dataset, fly_v31)
                           - "sandbox" (i.e. fly_v26)
                           - "public"
                           - "flat_630" or "flat_571" will use the flat
                             segmentations matching the respective materialization
                             versions. By default these use `lod=2`, you can
                             change that behaviour by passing `lod` as keyword
                             argument.

    Return
    ------
    skeleton :          navis.TreeNeuron
                        The extracted skeleton.

    See Also
    --------
    :func:`fafbseg.flywire.skeletonize_neuron_parallel`
                        Use this if you want to skeletonize many neurons in
                        parallel.
    :func:`fafbseg.flywire.get_l2_skeleton`
                        Generate a skeleton using the L2 cache. Much faster than
                        skeletonization from scratch but the skeleton will be
                        coarser.
    :func:`~fafbseg.flywire.get_skeletons`
                        Use this function to fetch precomputed skeletons. Only
                        available for proofread neurons and for specific
                        materialization versions.

    Examples
    --------
    >>> from fafbseg import flywire
    >>> n = flywire.skeletonize_neuron(720575940603231916)

    """
    if save_to is not None:
        save_to = pathlib.Path(save_to)
        if not save_to.exists():
            raise ValueError('`save_to` must be an existing directory')
        if not save_to.is_dir():
            raise ValueError('`save_to` must be a directory')

    # TODOs:
    # - drop single disconnected nodes?
    # - heal fragmented neurons?
    # - fix 0-radius nodes: these will be on 99.9% of the cases be leaf nodes
    # - shave only high-strahler twigs

    if int(sk.__version__.split('.')[0]) < 1:
        raise ImportError('Please update skeletor to version >= 1.0.0: '
                          'pip3 install skeletor -U')

    if navis.utils.is_iterable(x):
        # Make sure these are root IDs
        x = np.asarray(x).astype(np.int64)

        # Fetch all somas in one go (note that this will only find somas for
        # roots that actually existed at the time)
        # For neurons without a soma we'll be doing more sophisticated checks
        # when we skeletonize
        with silence_find_mat_version():
            kwargs['_nuclei'] = get_somas(x, raise_missing=False, dataset=dataset, materialization='latest')

        return navis.NeuronList([skeletonize_neuron(n,
                                                    progress=False,
                                                    shave_skeleton=shave_skeleton,
                                                    remove_soma_hairball=remove_soma_hairball,
                                                    assert_id_match=assert_id_match,
                                                    dataset=dataset,
                                                    threads=threads,
                                                    save_to=save_to,
                                                    **kwargs)
                                 for n in navis.config.tqdm(x,
                                                            desc='Skeletonizing',
                                                            disable=not progress,
                                                            leave=False)])

    if not navis.utils.is_mesh(x):
        vol = get_cloudvolume(dataset)

        # Make sure this is a valid integer
        id = np.int64(x)

        # Download the mesh
        try:
            old_parallel = vol.parallel
            vol.parallel = threads
            if vol.path.startswith('graphene'):
                mesh = vol.mesh.get(id, deduplicate_chunk_boundaries=False)[id]
            elif vol.path.startswith('precomputed'):
                lod_ = kwargs.pop('lod', 2)
                while lod_ >= 0:
                    try:
                        mesh = vol.mesh.get(id, lod=lod_)[id]
                        break
                    except cv.exceptions.MeshDecodeError:
                        lod_ -= 1
                    except BaseException:
                        raise
                if lod_ < 0:
                    raise ValueError(f'Root ID {id} does not appear to exist '
                                     f'in "{dataset}"')
        except BaseException:
            raise
        finally:
            vol.parallel = old_parallel
    else:
        mesh = x
        id = getattr(mesh, 'segid', getattr(mesh, 'id', 0))

    # Pop nuclei from kwargs before passing them to skeletonization
    nuc = kwargs.pop('_nuclei', pd.DataFrame())

    mesh = sk.utilities.make_trimesh(mesh, validate=True)

    # Fix things before we skeletonize
    # Drop disconnected pieces that represent less than 0.05% of total size
    to_remove = int(0.0001 * mesh.vertices.shape[0])
    to_remove = None if to_remove == 0 else to_remove
    mesh = sk.pre.fix_mesh(mesh,
                           inplace=True,
                           remove_disconnected=to_remove)

    # Skeletonize
    defaults = dict(waves=1, step_size=1)
    defaults.update(kwargs)
    s = sk.skeletonize.by_wavefront(mesh, progress=progress, **defaults)

    # Skeletor indexes node IDs at zero but to avoid potential issues we want
    # node IDs to start at 1
    s.swc['node_id'] += 1
    s.swc.loc[s.swc.parent_id >= 0, 'parent_id'] += 1

    # We will also round the radius and make it an integer to save some
    # memory. We could do the same with x/y/z coordinates but that could
    # potentially move nodes outside the mesh
    s.swc['radius'] = s.swc.radius.round().astype(int)

    # Turn into a neuron
    tn = navis.TreeNeuron(s.swc, units='1 nm', id=id, soma=None)

    if shave_skeleton:
        # Get child -> parent distances
        d = navis.morpho.mmetrics.parent_dist(tn, root_dist=0)
        # Find all nodes whose parent is more than a micron away (suspicious)
        long = tn.nodes[d >= 1000].node_id.values
        # Now start shaving
        while True:
            # Find segments containing leafs
            leaf_segs = [seg for seg in tn.small_segments if seg[0] in tn.leafs.node_id.values]
            # Among the leaf segments find those that are either only 1 node
            # or have any of the suspicously long (> micron) connections
            to_remove = [seg for seg in leaf_segs if any(np.isin(seg, long)) or (len(seg) <= 2)]

            # Make sure we don't drop very long segments
            to_remove = [seg for seg in to_remove if len(seg) < 10]

            # Turn list of lists into list of node IDs
            to_remove = [n for l in to_remove for n in l[:-1]]

            # If nothing more to remove, we can stop here
            if not len(to_remove):
                break

            navis.subset_neuron(tn, ~tn.nodes.node_id.isin(to_remove), inplace=True)

        # Get branch points
        bp = tn.nodes.loc[tn.nodes.type == 'branch', 'node_id'].values

        # Get single-node twigs
        is_end = tn.nodes.type == 'end'
        parent_is_bp = tn.nodes.parent_id.isin(bp)
        twigs = tn.nodes.loc[is_end & parent_is_bp, 'node_id'].values

        # Drop terminal twigs
        tn._nodes = tn.nodes.loc[~tn.nodes.node_id.isin(twigs)].copy()
        tn._clear_temp_attr()

    # If nuclei have already been fetched for all neurons
    if not nuc.empty:
        soma = nuc[nuc.pt_root_id == id]
    else:
        soma = pd.DataFrame()

    if soma.empty:
        # See if we can find a soma based on the nucleus segmentation
        try:
            with silence_find_mat_version():
                soma = get_somas(id,
                                 raise_missing=False,
                                 dataset=dataset,
                                 materialization='auto')
        except KeyboardInterrupt:
            raise
        except requests.HTTPError:
            navis.config.logger.warning(f'Failed to fetch soma for {id} from '
                                        'nucleus table.')
            soma = pd.DataFrame()

    if not soma.empty:
        soma = tn.snap(soma.iloc[0].pt_position)[0]
    else:
        # If no nucleus, try to detect the soma like this
        soma = detect_soma_skeleton(tn, min_rad=800, N=3)

    if soma:
        tn.soma = soma

        # Reroot to soma
        tn.reroot(tn.soma, inplace=True)

        if remove_soma_hairball:
            soma = tn.nodes.set_index('node_id').loc[soma]
            soma_loc = soma[['x', 'y', 'z']].values

            # Find all nodes within 2x the soma radius
            tree = navis.neuron2KDTree(tn)
            ix = tree.query_ball_point(soma_loc, max(4000, soma.radius * 2))

            # Translate indices into node IDs
            ids = tn.nodes.iloc[ix].node_id.values

            # Find segments that contain these nodes
            segs = [s for s in tn.segments if any(np.isin(ids, s))]

            # Sort segs by length
            segs = sorted(segs, key=lambda x: len(x))

            # Keep only the longest segment in that initial list
            to_drop = np.array([n for s in segs[:-1] for n in s])
            to_drop = to_drop[~np.isin(to_drop, segs[-1] + [soma.name])]

            navis.remove_nodes(tn, to_drop, inplace=True)

    if assert_id_match:
        if id == 0:
            raise ValueError('Segmentation ID must not be 0')
        new_locs = snap_to_id(tn.nodes[['x', 'y', 'z']].values,
                              id=id,
                              snap_zero=False,
                              dataset=dataset,
                              search_radius=160,
                              coordinates='nm',
                              max_workers=4,
                              verbose=True)
        tn.nodes[['x', 'y', 'z']] = new_locs

    if save_to is not None:
        navis.write_swc(tn, save_to / f'{tn.id}.swc')

    return tn


def detect_soma_skeleton(s, min_rad=800, N=3):
    """Try detecting the soma based on radii.

    Parameters
    ----------
    s :         navis.TreeNeuron
    min_rad :   float
                Minimum radius for a node to be considered a soma candidate.
    N :         int
                Number of consecutive nodes with radius > `min_rad` we need in
                order to consider them soma candidates.

    Returns
    -------
    node ID

    """
    assert isinstance(s, navis.TreeNeuron)

    # For each segment get the radius
    radii = s.nodes.set_index('node_id').radius.to_dict()
    candidates = []
    for seg in s.segments:
        rad = np.array([radii[s] for s in seg])
        is_big = np.where(rad > min_rad)[0]

        # Skip if no above-threshold radii in this segment
        if not any(is_big):
            continue

        # Find stretches of consectutive above-threshold radii
        for stretch in np.split(is_big, np.where(np.diff(is_big) != 1)[0]+1):
            if len(stretch) < N:
                continue
            candidates += [seg[i] for i in stretch]

    if not candidates:
        return None

    # Return largest candidate
    return sorted(candidates, key=lambda x: radii[x])[-1]


def __detect_soma_mesh(mesh):
    """Try detecting the soma based on vertex clusters.

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

    # Find out how many neighbours each vertex has within a 4 micron radius
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


def divide_local_neighbourhood(mesh, radius):
    """Divide the mesh into locally connected patches of a given size.

    All nodes will be assigned to a patches but patches will be overlapping.

    Parameters
    ----------
    mesh :      trimesh.Trimesh
    radius :    float

    Returns
    -------
    list of sets

    """
    assert isinstance(mesh, tm.Trimesh)
    assert isinstance(radius, numbers.Number)

    # Generate a graph for mesh
    G = mesh.vertex_adjacency_graph
    # Use Eucledian distance for edge weights
    edges = np.array(G.edges)
    e1 = mesh.vertices[edges[:, 0]]
    e2 = mesh.vertices[edges[:, 1]]
    dist = np.sqrt(np.sum((e1 - e2) ** 2, axis=1))
    nx.set_edge_attributes(G, dict(zip(G.edges, dist)), name='weight')

    not_seen = set(G.nodes)
    patches = []
    while not_seen:
        center = not_seen.pop()
        sg = nx.ego_graph(G, center, distance='weight', radius=radius)
        nodes = set(sg.nodes)
        patches.append(nodes)
        not_seen -= nodes


def skeletonize_neuron_parallel(ids, n_cores=os.cpu_count() // 2, progress=True, **kwargs):
    """Skeletonization on parallel cores.

    Parameters
    ----------
    ids :       iterable
                Root IDs of neurons you want to skeletonize.
    n_cores :   int
                Number of cores to use. Don't go too crazy on this as the
                downloading of meshes becomes a bottle neck if you try to do
                too many at the same time. Keep your internet speed in
                mind. For reference: with 100Mbps internet, I can comfortably
                run on 8 cores with some room to spare.
    **kwargs
                Keyword arguments are passed on to `skeletonize_neuron`.

    Returns
    -------
    navis.NeuronList

    See Also
    --------
    :func:`fafbseg.flywire.skeletonize_neuron`
                The function called for individual neurons.

    """
    if n_cores < 2 or n_cores > os.cpu_count():
        raise ValueError('`n_cores` must be between 2 and max number of cores.')

    sig = inspect.signature(skeletonize_neuron)
    for k in kwargs:
        if k not in sig.parameters and k not in ('lod', ):
            raise ValueError('unexpected keyword argument for '
                             f'`skeletonize_neuron`: {k}')

    # Make sure IDs are all integers
    ids = np.asarray(ids, dtype=np.int64)

    # Prepare the calls and parameters
    kwargs['progress'] = False
    kwargs['threads'] = 1
    kwargs['_nuclei'] = get_somas(ids, raise_missing=False, dataset=kwargs.get('dataset', 'production'))
    funcs = [skeletonize_neuron] * len(ids)
    parsed_kwargs = [kwargs] * len(ids)
    combinations = list(zip(funcs, [[i] for i in ids], parsed_kwargs))

    # Run the actual skeletonization
    with mp.Pool(n_cores) as pool:
        chunksize = 1
        res = list(navis.config.tqdm(pool.imap(_worker_wrapper,
                                               combinations,
                                               chunksize=chunksize),
                                     total=len(combinations),
                                     desc='Skeletonizing',
                                     disable=not progress,
                                     leave=True))

    # Check if any skeletonizations failed
    failed = np.array([r for r in res if not isinstance(r, navis.TreeNeuron)]).astype(str)
    if any(failed):
        print(f'{len(failed)} neurons failed to skeletonize: '
              f'{". ".join(failed)}')

    return navis.NeuronList([r for r in res if isinstance(r, navis.TreeNeuron)])


def _worker_wrapper(x):
    f, args, kwargs = x
    try:
        return f(*args, **kwargs)
    except KeyboardInterrupt:
        raise
    # We implement a single retry in case of HTTP errors
    except requests.HTTPError:
        try:
            return f(*args, **kwargs)
        except BaseException:
            # In case of failure return the root ID
            return args[0]
    except BaseException:
        # In case of failure return the root ID
        return args[0]


@parse_neuroncriteria()
def get_skeletons(root_id, threads=2, omit_failures=None, max_threads=6,
                  dataset=783, progress=True):
    """Fetch precomputed skeletons.

    Currently this only works for proofread (!) 630 and 783 root IDs
    (i.e. the first two public releases of FlyWire).

    Parameters
    ----------
    root_id  :          int | list of ints | NeuronCriteria
                        Root ID(s) of the FlyWire neuron(s) you want to
                        skeletonize. Must be root IDs that existed at
                        materialization 630 or 783 (see `dataset` parameter).
    omit_failures :     bool, optional
                        Determine behaviour when skeleton generation fails
                        (e.g. if the neuron has only a single chunk):
                         - ``None`` (default) will raise an exception
                         - ``True`` will skip the offending neuron (might result
                           in an empty ``NeuronList``)
                         - ``False`` will return an empty ``TreeNeuron``
    dataset :           630 | 783
                        Which dataset to query.
    progress :          bool
                        Whether to show a progress bar.
    max_threads :       int
                        Number of parallel requests to make when fetching the
                        skeletons.

    Returns
    -------
    skeletons :         navis.NeuronList | navis.TreeNeurons
                        Either a single neuron or a list thereof.

    See Also
    --------
    :func:`~fafbseg.flywire.skeletonize_neuron`
                        Use this function to skeletonize neurons from scratch,
                        e.g. if there aren't any precomputed skeletons
                        available.

    Examples
    --------
    >>> from fafbseg import flywire
    >>> n = flywire.get_skeletons(720575940603231916)
    >>> n                                                   #doctest: +SKIP
    type                                             navis.TreeNeuron
    name                                                     skeleton
    id                                             720575940603231916
    n_nodes                                                      3588
    n_connectors                                                 None
    n_branches                                                    586
    n_leafs                                                       645
    cable_length                                           2050971.75
    soma                                                         None
    units                                                 1 nanometer
    dtype: object

    """
    if str(dataset) not in SKELETON_BASE_URL:
        raise ValueError(
            "Currently we only provide precomputed skeletons for the "
            "630 and 783 data releases."
        )

    if omit_failures not in (None, True, False):
        raise ValueError(
            "`omit_failures` must be either None, True or False. "
            f'Got "{omit_failures}".'
        )

    if navis.utils.is_iterable(root_id):
        root_id = np.asarray(root_id, dtype=np.int64)

        il = is_latest_root(root_id, timestamp=f"mat_{dataset}")
        if np.any(~il):
            msg = (
                f"{(~il).sum()} root ID(s) did not exists at materialization {dataset}"
            )
            if omit_failures is None:
                raise ValueError(msg)
            navis.config.logger.warning(msg)

        get_skels = partial(get_skeletons, omit_failures=omit_failures, dataset=dataset)
        if (max_threads > 1) and (len(root_id) > 1):
            with ThreadPoolExecutor(max_workers=max_threads) as pool:
                futures = pool.map(get_skels, root_id)
                nl = [
                    f
                    for f in navis.config.tqdm(
                        futures,
                        desc="Fetching skeletons",
                        total=len(root_id),
                        disable=not progress or len(root_id) == 1,
                        leave=False,
                    )
                ]
        else:
            nl = [
                get_skels(r)
                for r in navis.config.tqdm(
                    root_id,
                    desc="Fetching skeletons",
                    total=len(root_id),
                    disable=not progress or len(root_id) == 1,
                    leave=False,
                )
            ]

        # Turn into neuron list
        nl = navis.NeuronList(nl)

        # Bring in original order
        if len(nl):
            root_id = root_id[np.isin(root_id, nl.id)]
            nl = nl.idx[root_id]

        return nl

    # Turn into integer
    root_id = np.int64(root_id)

    try:
        tn = navis.read_precomputed(f'{SKELETON_BASE_URL[str(dataset)]}/{root_id}',
                                    datatype='skeleton',
                                    info=SKELETON_INFO)
        # Force integer (navis.read_precomputed will turn Id into string)
        tn.id = root_id
        tn.units = '1nm'
        return tn
    except BaseException:
        if omit_failures is None:
            raise
        elif omit_failures:
            return navis.NeuronList([])
        else:
            return navis.TreeNeuron(None, id=root_id, units='1 nm')
