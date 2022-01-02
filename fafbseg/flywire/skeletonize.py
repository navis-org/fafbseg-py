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

import multiprocessing as mp
import networkx as nx
import numpy as np
import skeletor as sk
import trimesh as tm

from .segmentation import snap_to_id, is_latest_root
from .utils import parse_volume
from .meshes import detect_soma
from .annotations import get_somas


__all__ = ['skeletonize_neuron', 'skeletonize_neuron_parallel']


def skeletonize_neuron(x, shave_skeleton=True, remove_soma_hairball=False,
                       assert_id_match=False, dataset='production',
                       progress=True, **kwargs):
    """Skeletonize FlyWire neuron.

    Note that this is optimized to be primarily fast which comes at the cost
    of (some) quality. Also note that soma detection is using the nucleus
    segmentation and falls back to a radius-based heuristic if no nucleus is
    found.

    Parameters
    ----------
    x  :                 int | trimesh.TriMesh | list thereof
                         ID(s) or trimesh of the FlyWire neuron(s) you want to
                         skeletonize.
    shave_skeleton :     bool
                         If True, we will "shave" the skeleton by removing all
                         single-node terminal twigs. This should get rid of
                         bristles on the backbone that can occur if the neurites
                         are very big.
    remove_soma_hairball : bool
                         If True, we will try to drop the hairball that is
                         typically created inside the soma. Note that while this
                         should work just fine for 99% of neurons, it's not very
                         smart and there is always a small chance that we
                         remove stuff that should not have been removed. Also
                         only works if the neuron has it's nucleus annotated
                         (see :func:`fafbseg.flywire.get_somas`).
    assert_id_match :    bool
                         If True, will check if skeleton nodes map to the
                         correct segment ID and if not will move them back into
                         the segment. This is potentially very slow!
    dataset :            str | CloudVolume
                         Against which FlyWire dataset to query::
                           - "production" (current production dataset, fly_v31)
                           - "sandbox" (i.e. fly_v26)
    progress :           bool
                         Whether to show a progress bar or not.

    Return
    ------
    skeleton :          navis.TreeNeuron
                        The extracted skeleton.

    See Also
    --------
    :func:`fafbseg.flywire.skeletonize_neuron_parallel`
                        Use this if you want to skeletonize many neurons in
                        parallel.

    Examples
    --------
    >>> from fafbseg import flywire
    >>> n = flywire.skeletonize_neuron(720575940614131061)

    """

    if int(sk.__version__.split('.')[0]) < 1:
        raise ImportError('Please update skeletor to version >= 1.0.0: '
                          'pip3 install skeletor -U')

    vol = parse_volume(dataset)

    if navis.utils.is_iterable(x):
        return navis.NeuronList([skeletonize_neuron(n,
                                                    progress=False,
                                                    remove_soma_hairball=remove_soma_hairball,
                                                    assert_id_match=assert_id_match,
                                                    dataset=dataset,
                                                    **kwargs)
                                 for n in navis.config.tqdm(x,
                                                            desc='Skeletonizing',
                                                            disable=not progress,
                                                            leave=False)])

    if not navis.utils.is_mesh(x):
        vol = parse_volume(dataset)

        # Make sure this is a valid integer
        id = int(x)

        # Download the mesh
        mesh = vol.mesh.get(id, deduplicate_chunk_boundaries=False,
                            remove_duplicate_vertices=True)[id]
    else:
        mesh = x
        id = getattr(mesh, 'segid', getattr(mesh, 'id', 0))

    mesh = sk.utilities.make_trimesh(mesh, validate=False)

    # Fix things before we skeletonize
    # This also drops fluff
    mesh = sk.pre.fix_mesh(mesh, inplace=True, remove_disconnected=100)

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
        # Get branch points
        bp = tn.nodes.loc[tn.nodes.type == 'branch', 'node_id'].values

        # Get single-node twigs
        is_end = tn.nodes.type == 'end'
        parent_is_bp = tn.nodes.parent_id.isin(bp)
        twigs = tn.nodes.loc[is_end & parent_is_bp, 'node_id'].values

        # Drop terminal twigs
        tn._nodes = tn.nodes.loc[~tn.nodes.node_id.isin(twigs)].copy()
        tn._clear_temp_attr()

    # See if we can find a soma based on the nucleus segmentation
    if is_latest_root(id):
        materialization = 'live'
    else:
        materialization = 'latest'
    soma = get_somas(id, dataset=dataset, materialization=materialization)
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


def skeletonize_neuron_parallel(ids, n_cores=os.cpu_count() // 2, **kwargs):
    """Skeletonization on parallel cores [WIP].

    Parameters
    ----------
    ids :       iterable
                Root IDs of neurons you want to skeletonize.
    n_cores :   int
                Number of cores to use. Don't go too crazy on this as the
                downloading of meshes becomes a bottle neck if you try to do
                too many at the same time. Keep your internet speed in
                mind.
    **kwargs
                Keyword arguments are passed on to `skeletonize_neuron`.

    Returns
    -------
    navis.NeuronList

    """
    if n_cores < 2 or n_cores > os.cpu_count():
        raise ValueError('`n_cores` must be between 2 and max number of cores.')

    sig = inspect.signature(skeletonize_neuron)
    for k in kwargs:
        if k not in sig.parameters:
            raise ValueError('unexpected keyword argument for '
                             f'`skeletonize_neuron`: {k}')

    # Make sure IDs are all integers
    ids = np.asarray(ids).astype(int)

    # Prepare the calls and parameters
    kwargs['progress'] = False
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
                                     disable=False,
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
