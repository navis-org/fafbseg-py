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

import itertools
import pymaid
import navis
import scipy.spatial

import networkx as nx
import numpy as np

from .. import utils

# This is to prevent FutureWarning from numpy (via vispy)
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

use_pbars = utils.use_pbars


def collapse_nodes(A, B, limit=1, base_neuron=None, mesh=None):
    """Merge neuron A into neuron(s) B creating a union of both.

    This implementation uses edge contraction on the neurons' graph to ensure
    maximum connectivity. Only works if the fragments collectively form a
    continuous tree (i.e. you must be certain that they partially overlap).

    Parameters
    ----------
    A :                 CatmaidNeuron
                        Neuron to be collapsed into neurons B.
    B :                 CatmaidNeuronList
                        Neurons to collapse neuron A into.
    limit :             int, optional
                        Max distance [microns] for nearest neighbour search.
    base_neuron :       skeleton ID | CatmaidNeuron, optional
                        Neuron from B to use as template for union. If not
                        provided, the first neuron in the list is used as
                        template!
    mesh :              navis.Volume | navis.MeshNeuron | mesh-like object
                        If provided, will use the mesh to check if nodes are
                        in line of sight to each other before collapsing them.

    Returns
    -------
    core.CatmaidNeuron
                        Union of all input neurons.
    new_edges :         pandas.DataFrame
                        Subset of the ``.nodes`` table that represent newly
                        added edges.
    collapsed_nodes :   dict
                        Map of collapsed nodes::

                            NodeA -collapsed-into-> NodeB

    """
    if isinstance(A, navis.NeuronList):
        if len(A) == 1:
            A = A[0]
        else:
            A = navis.stitch_neurons(A, method="NONE")

    if not isinstance(A, navis.TreeNeuron):
        raise TypeError('`A` must be a TreeNeuron, got "{}"'.format(type(A)))

    if isinstance(B, navis.TreeNeuron):
        B = navis.NeuronList(B)

    if not isinstance(B, navis.NeuronList):
        raise TypeError('`B` must be a NeuronList, got "{}"'.format(type(B)))

    if isinstance(base_neuron, type(None)):
        base_neuron = B[0]

    # This is just check on the off-chance that skeleton IDs are not unique
    # (e.g. if neurons come from different projects) -> this is relevant because
    # we identify the master ("base_neuron") via it's skeleton ID
    skids = [n.id for n in B + A]
    if len(skids) > len(set(skids)):
        raise ValueError('Duplicate skeleton IDs found. Try manually assigning '
                         'unique skeleton IDs.')

    # Convert distance threshold from microns to nanometres
    limit *= 1000

    # Before we start messing around, let's make sure we can keep track of
    # the origin of each node
    for n in B + A:
        n.nodes['origin_skeletons'] = n.id

    # First make a weak union by simply combining the node tables
    B.neurons = sorted(B.neurons, key=lambda x: {base_neuron: 0}.get(x, 2))
    union_simple = navis.stitch_neurons(B + A, method='NONE', master='FIRST')

    # Check for duplicate node IDs
    if any(union_simple.nodes.node_id.duplicated()):
        raise ValueError('Duplicate node IDs found.')

    # Find nodes in A to be merged into B
    tree = scipy.spatial.cKDTree(data=B.nodes[['x', 'y', 'z']].values)

    # For each node in A get the nearest neighbor in B
    coords = A.nodes[['x', 'y', 'z']].values
    nn_dist, nn_ix = tree.query(coords, k=1, distance_upper_bound=limit)

    # Find nodes that are close enough to collapse
    collapsed = A.nodes.loc[nn_dist <= limit].node_id.values
    clps_into = B.nodes.iloc[nn_ix[nn_dist <= limit]].node_id.values

    # If we have a mesh, check if those collapsed nodes are in sight of each
    # other
    if mesh:
        import ncollpyde
        coll = ncollpyde.Volume(mesh.vertices, mesh.faces)

        # Produce start and end coordinates for the to collapse nodes
        starts = A.nodes.set_index('node_id').loc[collapsed,
                                                  ['x', 'y', 'z']].values
        ends = B.nodes.set_index('node_id').loc[clps_into,
                                                ['x', 'y', 'z']].values

        # Check if the line between start and end intersects the mesh
        intersects, _, _ = coll.intersections(starts, ends)

        # Indices that show up in `intersects` cross a membrane
        # -> we need to invert this to get those nodes that don't
        not_intersects = ~np.isin(np.arange(starts.shape[0]), intersects)

        # Keep only collapses that don't intersect
        collapsed = collapsed[not_intersects]
        clps_into = clps_into[not_intersects]

    # Generate a map of which node in A is to be collapsed into which node in B
    clps_map = {n1: n2 for n1, n2 in zip(collapsed, clps_into)}

    # The fastest way to collapse is to work on the edge list
    E = nx.to_pandas_edgelist(union_simple.graph)

    # Keep track of which edges were collapsed -> we will use this as weight
    # later on to prioritize existing edges over newly generated ones
    E['is_new'] = 1
    source_in_B = E.source.isin(B.nodes.node_id.values)
    target_in_B = E.target.isin(B.nodes.node_id.values)
    E.loc[source_in_B | target_in_B, 'is_new'] = 0

    # Now map collapsed nodes onto the nodes they collapsed into
    E['target'] = E.target.map(lambda x: clps_map.get(x, x))
    E['source'] = E.source.map(lambda x: clps_map.get(x, x))

    # Make sure no self loops after collapsing. This happens if two adjacent
    # nodes collapse onto the same target node
    E = E[E.source != E.target]

    # Remove duplicates. This happens e.g. when two adjaceny nodes merge into
    # two other adjaceny nodes: A->B C->D ----> A/B->C/D
    # By sorting first, we make sure original edges are kept first
    E.sort_values('is_new', ascending=True, inplace=True)

    # Because edges may exist in both directions (A->B and A<-B) we have to
    # generate a column that's agnostic to directionality using frozensets
    E['frozen_edge'] = E[['source', 'target']].apply(frozenset, axis=1)
    E.drop_duplicates(['frozen_edge'], keep='first', inplace=True)

    # Regenerate graph from these new edges
    G = nx.Graph()
    G.add_weighted_edges_from(E[['source', 'target', 'is_new']].values.astype(int))

    # At this point there might still be disconnected pieces -> we will create
    # separate neurons for each tree
    props = union_simple.nodes.loc[union_simple.nodes.node_id.isin(G.nodes)].set_index('node_id')
    nx.set_node_attributes(G, props.to_dict(orient='index'))
    fragments = []
    for n in nx.connected_components(G):
        c = G.subgraph(n)
        tree = nx.minimum_spanning_tree(c)
        fragments.append(navis.graph.nx2neuron(tree,
                                               name=base_neuron.name,
                                               id=base_neuron.id))
    fragments = navis.NeuronList(fragments)

    if len(fragments) > 1:
        print('Union incomplete - watch out for disconnected fragments!')
        # Now heal those fragments using a minimum spanning tree
        union = navis.stitch_neurons(*fragments, method='ALL')
    else:
        union = fragments[0]

    # Reroot to base neuron's root
    union.reroot(base_neuron.root[0], inplace=True)

    # Add tags back on
    if union_simple.has_tags:
        if not union.has_tags:
            union.tags = {}
        union.tags.update(union_simple.tags)

    # Add connectors back on
    union.connectors = union_simple.connectors.drop_duplicates(subset='connector_id').copy()
    union.connectors.loc[:, 'node_id'] = union.connectors.node_id.map(lambda x: clps_map.get(x, x))

    # Find the newly added edges (existing edges should not have been modified
    # - except for changing direction due to reroot)
    # The basic logic here is that new edges were only added between two
    # previously separate skeletons, i.e. where the skeleton ID changes between
    # parent and child node
    node2skid = union_simple.nodes.set_index('node_id').origin_skeletons.to_dict()
    union.nodes['parent_skeleton'] = union.nodes.parent_id.map(node2skid)
    new_edges = union.nodes[union.nodes.origin_skeletons != union.nodes.parent_skeleton]
    # Remove root edges
    new_edges = new_edges[~new_edges.parent_id.isnull()]

    return union, new_edges, clps_map


def __collapse_nodes(*x, limit=1, base_neuron=None, priority_nodes=None):
    """Generate the union of a set of neurons.

    This implementation uses edge contraction on the neurons' graph to ensure
    maximum connectivity. Only works if, taken together, the neurons form a
    continuous tree (i.e. you must be certain that they partially overlap).

    Parameters
    ----------
    *x :                CatmaidNeuron/List
                        Neurons to be merged.
    limit :             int, optional
                        Max distance [microns] for nearest neighbour search.
    base_neuron :       skeleton_ID | CatmaidNeuron, optional
                        Neuron to use as template for union. If not provided,
                        the first neuron in the list is used as template!
    priority_nodes :    list-like
                        List of treenode IDs. If provided, these nodes will
                        have priority when pairwise collapsing nodes. If two
                        priority nodes are to be collapsed, a new edge between
                        them is created instead.

    Returns
    -------
    core.CatmaidNeuron
                        Union of all input neurons.
    collapsed_nodes :   dict
                        Map of collapsed nodes::

                            NodeA -collapsed-into-> NodeB

    new_edges :         list
                        List of newly added edges::

                            [[NodeA, NodeB], ...]

    """
    # Unpack neurons in *args
    x = pymaid.utils._unpack_neurons(x)

    # Make sure we're working on copies and don't change originals
    x = pymaid.CatmaidNeuronList([n.copy() for n in x])

    if isinstance(priority_nodes, type(None)):
        priority_nodes = []

    # This is just check on the off-chance that skeleton IDs are not unique
    # (e.g. if neurons come from different projects) -> this is relevant because
    # we identify the master ("base_neuron") via it's skeleton ID
    skids = [n.id for n in x]
    if len(skids) > len(set(skids)):
        raise ValueError('Duplicate skeleton IDs found in neurons to be merged. '
                         'Try manually assigning unique skeleton IDs.')

    if any([not isinstance(n, pymaid.CatmaidNeuron) for n in x]):
        raise TypeError('Input must only be CatmaidNeurons/List')

    if len(x) < 2:
        raise ValueError('Need at least 2 neurons to make a union!')

    # Convert distance threshold from microns to nanometres
    limit *= 1000

    # First make a weak union by simply combining the node tables
    union_simple = pymaid.stitch_neurons(x, method='NONE', master=base_neuron)

    # Check for duplicate node IDs
    if any(union_simple.nodes.node_id.duplicated()):
        raise ValueError('Duplicate node IDs found.')

    # Map priority nodes -> this will speed things up later
    is_priority = {n: True for n in priority_nodes}

    # Go over each pair of fragments and check if they can be collapsed
    comb = itertools.combinations(x, 2)
    collapse_into = {}
    new_edges = []
    for c in comb:
        tree = pymaid.neuron2KDTree(c[0], tree_type='c', data='nodes')

        # For each node in master get the nearest neighbor in minion
        coords = c[1].nodes[['x', 'y', 'z']].values
        nn_dist, nn_ix = tree.query(coords, k=1, distance_upper_bound=limit)

        clps_left = c[0].nodes.iloc[nn_ix[nn_dist <= limit]].node_id.values
        clps_right = c[1].nodes.iloc[nn_dist <= limit].node_id.values
        clps_dist = nn_dist[nn_dist <= limit]

        for i, (n1, n2, d) in enumerate(zip(clps_left, clps_right, clps_dist)):
            if is_priority.get(n1, False):
                # If both nodes are priority nodes, don't collapse
                if is_priority.get(n2, False):
                    new_edges.append([n1, n2, d])
                    # continue
                else:
                    collapse_into[n2] = n1
            else:
                collapse_into[n1] = n2

    # Get the graph
    G = union_simple.graph

    # Add the new edges to graph
    G.add_weighted_edges_from(new_edges)

    # Using an edge list is much more efficient than an adjacency matrix
    E = nx.to_pandas_edgelist(G)

    # All nodes that collapse into other nodes need to have weight set to
    # float("inf") to de-prioritize them when generating the minimum spanning
    # tree later
    clps_nodes = set(collapse_into.keys())
    E.loc[(E.source.isin(clps_nodes)) | (E.target.isin(clps_nodes)), 'weight'] = float('inf')

    # Now map collapsed nodes onto the nodes they collapsed into
    E['target'] = E.target.map(lambda x: collapse_into.get(x, x))
    E['source'] = E.source.map(lambda x: collapse_into.get(x, x))

    # Make sure no self loops after collapsing. This happens if two adjacent
    # nodes collapse onto the same target node
    E = E[E.source != E.target]

    # Turn this back into a graph
    G_clps = nx.from_pandas_edgelist(E, edge_attr='weight')

    # Make sure that we are fully connected
    if not nx.is_connected(G_clps):
        raise ValueError('Neuron still fragmented after collapsing nodes. '
                         'Try increasing the `limit` parameter.')

    # Under certain conditions, collapsing nodes will introduce cycles:
    # Consider for example a graph: A->B->C D->E->F
    # Collapsing A and C into D will create a loop between B<->D
    # To fix this we have to create a minimum spanning tree.
    # In doing so, we need to prioritize existing edges over new edges
    # otherwise we would have to cut existing neurons -> this is why we set
    # weight of new edges to float("inf") earlier on

    # Generate the tree
    tree = nx.minimum_spanning_tree(G_clps.to_undirected(as_view=True))

    # Add properties to nodes
    survivors = np.unique(E[['source', 'target']])
    props = union_simple.nodes.set_index('node_id').loc[survivors]
    nx.set_node_attributes(tree, props.to_dict(orient='index'))

    # Recreate neuron
    union = pymaid.graph.nx2neuron(tree,
                                   neuron_name=union_simple.neuron_name,
                                   skeleton_id=union_simple.skeleton_id)

    # Add tags back on
    for n in x:
        if not n.has_tags:
            continue
        union.tags.update({k: union.tags.get(k, []) + [collapse_into.get(a, a) for a in v] for k, v in n.tags.items()})

    # Add connectors back on
    union.connectors = x.connectors.drop_duplicates(subset='connector_id')
    union.connectors.node_id = union.connectors.node_id.map(lambda x: collapse_into.get(x, x))

    # Return the last survivor
    return union, collapse_into, new_edges
