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

import networkx as nx
import numpy as np
import trimesh as tm

from .segmentation import snap_to_id
from .utils import parse_volume

try:
    import skeletor as sk
except ImportError:
    sk = None
except BaseException:
    raise

__all__ = ['skeletonize_neuron']


def skeletonize_neuron(x, drop_soma_hairball=False, contraction_kws={},
                       skeletonization_kws={}, radius_kws={},
                       assert_id_match=False, dataset='production'):
    """Skeletonize flywire neuron.

    Parameters
    ----------
    x  :                 int | trimesh.TriMesh
                         ID or trimesh of the flywire neuron you want to
                         skeletonize.
    drop_soma_hairball : bool
                         If True, we will try to drop the hairball that is
                         typically created inside the soma.
    contraction_kws :    dict
                         Optional parameters for the contraction phase. See
                         ``skeletor.contract``.
    skeletonization_kws : dict
                         Optional parameters for the skeletonization phase. See
                         ``skeletor.skeletonize``.
    radius_kws :         dict
                         Optional parameters for the radius extraction phase.
                         See ``skeletor.radius``.
    assert_id_match :    bool
                         If True, will check if skeleton nodes map to the
                         correct segment ID and if not will move them back into
                         the segment. This is potentially very slow!
    dataset :            str | CloudVolume
                         Against which flywire dataset to query::
                           - "production" (current production dataset, fly_v31)
                           - "sandbox" (i.e. fly_v26)

    Return
    ------
    skeleton, simpified_mesh, contracted_mesh
                        The extraced skeleton, simplified and contracted mesh,
                        respectively.

    Examples
    --------
    >>> tn, simp, cntr = flywire.skeletonize_flywire_neuron(720575940614131061)

    """
    if not sk:
        raise ImportError('Must install skeletor: pip3 install skeletor')

    if not navis.utils.is_mesh(x):
        vol = parse_volume(dataset)

        # Make sure this is a valid integer
        id = int(x)

        # Download the mesh
        mesh = vol.mesh.get(id, deduplicate_chunk_boundaries=False)[id]
    else:
        mesh = x
        id = getattr(mesh, 'segid', 0)

    # Simplify
    simp = sk.simplify(mesh, ratio=.2)

    # Validate before we detect the soma verts
    simp = sk.utilities.fix_mesh(simp, inplace=True)

    # Try detecting the soma
    if drop_soma_hairball:
        soma_verts = detect_soma(simp)

    # Contract
    defaults = dict(SL=40, WH0=2, epsilon=0.1, precision=1e-7, validate=False)
    defaults.update(contraction_kws)
    cntr = sk.contract(simp, **defaults)

    # Generate skeleton
    defaults = dict(method='vertex_clusters', sampling_dist=200, vertex_map=True,
                    validate=False)
    defaults.update(skeletonization_kws)
    swc = sk.skeletonize(cntr, **defaults)

    # Clean up
    cleaned = sk.clean(swc, mesh=mesh, validate=False)

    # Extract radii
    defaults = dict(validate=False)
    defaults.update(radius_kws)
    cleaned['radius'] = sk.radii(cleaned, mesh=mesh, **defaults)

    # Convert to neuron
    n = navis.TreeNeuron(cleaned, id=id, units='nm', soma=None)

    # Drop any nodes that are soma vertices
    if drop_soma_hairball and soma_verts.shape[0] > 0:
        keep = n.nodes.loc[~n.nodes.vertex_id.isin(soma_verts),
                           'node_id'].values
        n = navis.subset_neuron(n, keep)

    if assert_id_match:
        if id == 0:
            raise ValueError('Segmentation ID must not be 0')
        new_locs = snap_to_id(n.nodes[['x', 'y', 'z']].values,
                              id=id,
                              snap_zero=False,
                              dataset=dataset,
                              search_radius=160,
                              coordinates='nm',
                              max_workers=4,
                              verbose=True)
        n.nodes[['x', 'y', 'z']] = new_locs

    return (n, simp, cntr)


def detect_soma(mesh):
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
