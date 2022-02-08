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

"""Functions to work with synapse data from services.itanna.io"""

import navis

import networkx as nx
import numpy as np
import pandas as pd

from scipy.spatial import cKDTree
from tqdm.auto import tqdm

from .utils import assign_connectors
from .. import google, spine

conn = None


__all__ = ['get_neuron_synapses', 'get_neuron_connections']


def get_neuron_synapses(x, pre=True, post=True, collapse_connectors=False,
                        score_thresh=30, ol_thresh=2, dist_thresh=1000,
                        attach=True, drop_autapses=True, drop_duplicates=True,
                        db=None, verbose=True, ret='catmaid', progress=True):
    """Fetch synapses for a given neuron.

    Works by:
     1. Fetch segment IDs corresponding to a neuron
     2. Fetch Buhmann et al. synapses associated with these segment IDs
     3. Do some clean-up. See parameters for details.

    Notes
    -----
    1. If ``collapse_connectors=False`` the ``connector_id`` column is a
       simple enumeration and is effectively meaningless.
    2. The x/y/z coordinates always correspond to the presynaptic site (like
       with CATMAID connectors). These columns are renamed from "pre_{x|y|z}"
       in the database.
    3. Some of the clean-up assumes that the query neurons are unique neurons
       and not fragments of the same neuron. If that is the case, you might be
       better of running this function for each fragment individually.

    Parameters
    ----------
    x :             navis.Neuron | pymaid.CatmaidNeuron | NeuronList | int | list thereof
                    Neurons to fetch synapses for.
    pre/post :      bool
                    Whether to fetch pre- and/or postsynapses.
    collapse_connectors : bool
                    If True, we will pool presynaptic connections into
                    CATMAID-like connectors. If False each row represents a
                    connections.
    ol_thresh :     int, optional
                    If provided, will required a minimum node overlap between
                    a neuron and a segment for that segment to be included
                    (see step 1 above).
    score_thresh :  int, optional
                    If provided will only return synapses with a cleft score of
                    this or higher.
    dist_thresh :   int
                    Synapses further away than the given distance [nm] from any
                    node in the neuron than given distance will be discarded.
                    This is always with respect to the cleft site
                    irrespective of whether our neuron is pre- or postsynaptic
                    to it.
    attach :        bool
                    If True, will attach synapses as `.connectors` to neurons.
                    If False, will return DataFrame with synapses.
    drop_autapses : bool
                    If True, will drop synapses where pre- and postsynapse point
                    to the same neuron. Autapses are wrong in 99.9% of all cases
                    we've seen.
    drop_duplicates :   bool
                        If True, will merge synapses which connect the same
                        pair of pre-post segmentation IDs and are within
                        less than 2500nm.
    db :            str, optional
                    Must point to SQL database containing the synapse data. If
                    not provided will look for a `BUHMANN_SYNAPSE_DB`
                    environment variable.
    ret :           "catmaid" | "brief" | "full"
                    If "full" will return all synapse properties. If "brief"
                    will return more relevant subset. If "catmaid" will return
                    only CATMAID-like columns.
    progress :      bool
                    Whether to show progress bars or not.

    Return
    ------
    pd.DataFrame
                    Only if ``attach=False``.

    """
    # This is just so we throw a DB exception early and not wait for fetching
    # segment Ids first
    _ = get_connection(db)

    assert isinstance(x, (navis.BaseNeuron, navis.NeuronList))
    assert ret in ("catmaid", "brief", "full")

    if not isinstance(x, navis.NeuronList):
        x = navis.NeuronList([x])

    # Get segments for this neuron(s)
    seg_ids = google.neuron_to_segments(x)

    # Drop segments with overlap below threshold
    if ol_thresh:
        seg_ids = seg_ids.loc[seg_ids.max(axis=1) >= ol_thresh]

    # We will make sure that every segment ID is only attributed to a single
    # neuron
    not_top = seg_ids.values != seg_ids.max(axis=1).values.reshape((seg_ids.shape[0], 1))
    where_not_top = np.where(not_top)
    if np.any(where_not_top[0]):
        seg_ids.values[where_not_top] = None  # do not change this to 0

    # Fetch pre- and postsynapses associated with these segments
    # It's cheaper to get them all in one go
    syn = query_synapses(seg_ids.index.values,
                         score_thresh=score_thresh,
                         ret=ret if ret != 'catmaid' else 'brief',
                         db=None)

    # Drop autapses - they are most likely wrong
    if drop_autapses:
        syn = syn[syn.segmentid_pre != syn.segmentid_post]

    if drop_duplicates:
        dupl_thresh = 250
        # Deal with pre- and postsynapses separately
        while True:
            # Generate pairs of pre- and postsynaptic coordinates that are
            # suspiciously close
            pre_tree = cKDTree(syn[['pre_x', 'pre_y', 'pre_z']].values)
            post_tree = cKDTree(syn[['post_x', 'post_y', 'post_z']].values)
            pre_pairs = pre_tree.query_pairs(r=dupl_thresh)
            post_pairs = post_tree.query_pairs(r=dupl_thresh)

            # We will consider pairs for removal where both pre- OR postsynapse
            # are close - this is easy to change via below operator
            pairs = pre_pairs | post_pairs  # union of both
            pairs = np.array(list(pairs))

            # For each pair check if they connect the same IDs
            same_pre = syn.iloc[pairs[:, 0]].segmentid_pre.values == syn.iloc[pairs[:, 1]].segmentid_pre.values
            same_post = syn.iloc[pairs[:, 0]].segmentid_post.values == syn.iloc[pairs[:, 1]].segmentid_post.values
            same_cn = same_pre & same_post

            # If no more pairs to collapse break
            if same_cn.sum() == 0:
                break

            # Generate a graph from pairs
            G = nx.Graph()
            G.add_edges_from(pairs[same_cn])

            # Find the minimum number of nodes we need to remove
            # to separate the connectors
            to_rm = []
            for cn in nx.connected_components(G):
                to_rm += list(nx.minimum_node_cut(nx.subgraph(G, cn)))

            # Drop those nodes
            syn = syn.drop(index=syn.index.values[to_rm])
        # Reset index
        syn.reset_index(drop=True, inplace=True)

    if collapse_connectors:
        assign_connectors(syn)
    else:
        # Make fake IDs
        syn['connector_id'] = np.arange(syn.shape[0]).astype(np.int32)

    # Now associate synapses with neurons
    tables = []
    for c in tqdm(seg_ids.columns,
                  desc='Proc. neurons',
                  disable=not progress or seg_ids.shape[1] == 1,
                  leave=False):
        this_segs = seg_ids.loc[seg_ids[c].notnull(), c]
        is_pre = syn.segmentid_pre.isin(this_segs.index.values)
        is_post = syn.segmentid_post.isin(this_segs.index.values)

        # At this point we might see the exact same connection showing up in
        # `is_pre` and in `is_post`. This happens when we mapped both the
        # pre- and the postsynaptic segment to this neuron - likely an error.
        # In these cases we have to decide whether our neuron is truely pre-
        # or postsynaptic. For this we will use the overlap counts:
        # First find connections that would show up twice
        is_dupl = is_pre & is_post
        if any(is_dupl):
            dupl = syn[is_dupl]
            # Next get the overlap counts for the pre- and postsynaptic seg IDs
            dupl_pre_ol = seg_ids.loc[dupl.segmentid_pre, c].values
            dupl_post_ol = seg_ids.loc[dupl.segmentid_post, c].values

            # We go for the one with more overlap
            true_pre = dupl_pre_ol > dupl_post_ol

            # Propagate that decision
            is_pre[is_dupl] = true_pre
            is_post[is_dupl] = ~true_pre

        # Now get our synapses
        this_pre = syn[is_pre]
        this_post = syn[is_post]

        # Keep only one connector per presynapse
        # -> just like in CATMAID connector tables
        # Postsynaptic connectors will still show up multiple times
        if collapse_connectors:
            this_pre = this_pre.drop_duplicates('connector_id')

        # Combine pre- and postsynapses and keep track of the type
        connectors = pd.concat([this_pre, this_post], axis=0).reset_index(drop=True)
        connectors['type'] = 'post'
        connectors.iloc[:this_pre.shape[0],
                        connectors.columns.get_loc('type')] = 'pre'
        connectors['type'] = connectors['type'].astype('category')

        # Rename columns such that x/y/z corresponds to presynaptic sites
        connectors.rename({'pre_x': 'x', 'pre_y': 'y', 'pre_z': 'z'},
                          axis=1, inplace=True)

        # For CATMAID-like connector tables subset to relevant columns
        if ret == 'catmaid':
            connectors = connectors[['connector_id', 'x', 'y', 'z',
                                     'cleft_scores', 'type']].copy()

        # Map connectors to nodes
        # Note that this is where we enforce `dist_thresh`
        neuron = x.idx[c]
        tree = navis.neuron2KDTree(neuron)
        dist, ix = tree.query(connectors[['x', 'y', 'z']].values,
                              distance_upper_bound=dist_thresh)

        # Drop far away connectors
        connectors = connectors.loc[dist < np.inf]

        # Assign node IDs
        connectors['node_id'] = neuron.nodes.iloc[ix[dist < np.inf]].node_id.values

        # Somas can end up having synapses, which we know is wrong and is
        # relatively easy to fix
        if np.any(neuron.soma):
            somata = navis.utils.make_iterable(neuron.soma)
            s_locs = neuron.nodes.loc[neuron.nodes.node_id.isin(somata),
                                      ['x', 'y', 'z']].values
            # Find all nodes within 2 micron around the somas
            soma_node_ix = tree.query_ball_point(s_locs, r=2000)
            soma_node_ix = [n for l in soma_node_ix for n in l]
            soma_node_id = neuron.nodes.iloc[soma_node_ix].node_id.values

            # Drop connectors attached to these soma nodes
            connectors = connectors[~connectors.node_id.isin(soma_node_id)]

        if attach:
            neuron.connectors = connectors.reset_index(drop=True)
        else:
            connectors['neuron'] = neuron.id  # do NOT change the type of this
            tables.append(connectors)

    if not attach:
        connectors = pd.concat(tables, axis=0, sort=True).reset_index(drop=True)
        return connectors


def get_neuron_connections(sources, targets=None, agglomerate=True,
                           score_thresh=30, ol_thresh=5, dist_thresh=2000,
                           drop_duplicates=True, drop_autapses=True, db=None,
                           verbose=True):
    """Fetch connections between sets of neurons.

    Works by:
     1. Fetch segment IDs corresponding to given neurons
     2. Fetch Buhmann et al. synaptic connections between them
     3. Do some clean-up. See parameters for details - defaults are those used
        by Buhmann et al

    Parameters
    ----------
    sources :       navis.Neuron | pymaid.CatmaidNeuron | NeuronList
                    Presynaptic neurons to fetch connections for.
    targets :       navis.Neuron | pymaid.CatmaidNeuron | NeuronList | None
                    Postsynaptic neurons to fetch connections for. If ``None``,
                    ``targets = sources``.
    agglomerate :   bool
                    If True, will agglomerate connectivity by ID and return a
                    weighted edge list. If False, will return a list of
                    individual synapses.
    ol_thresh :     int, optional
                    If provided, will required a minimum node overlap between
                    a neuron and a segment for that segment to be included
                    (see step 1 above).
    score_thresh :  int, optional
                    If provided will only return synapses with a cleft score of
                    this or higher.
    dist_thresh :   int
                    Synapses further away than the given distance [nm] from any
                    node in the neuron than given distance will be discarded.
    drop_duplicates :   bool
                        If True, will merge synapses which connect the same
                        pair of pre-post segmentation IDs and are within
                        less than 250nm.
    drop_autapses :     bool
                        If True, will automatically drop autapses.
    db :            str, optional
                    Must point to SQL database containing the synapse data. If
                    not provided will look for a ```BUHMANN_SYNAPSE_DB``
                    environment variable.

    Return
    ------
    pd.DataFrame
                    Either edge list or list of synapses - see ``agglomerate``
                    parameter.

    """
    # This is just so we throw a DB exception early and not wait for fetching
    # segment Ids first
    _ = get_connection(db)

    assert isinstance(sources, (navis.BaseNeuron, navis.NeuronList))

    if isinstance(targets, type(None)):
        targets = sources

    assert isinstance(targets, (navis.BaseNeuron, navis.NeuronList))

    if not isinstance(sources, navis.NeuronList):
        sources = navis.NeuronList([sources])
    if not isinstance(targets, navis.NeuronList):
        targets = navis.NeuronList([targets])

    # Get segments for this neuron(s)
    unique_neurons = (sources + targets).remove_duplicates(key='id')
    seg_ids = google.neuron_to_segments(unique_neurons)

    # Drop segments with overlap below threshold
    if ol_thresh:
        seg_ids = seg_ids.loc[seg_ids.max(axis=1) >= ol_thresh]

    # We need to make sure that every segment ID is only attributed to a single
    # neuron
    is_top = seg_ids.values != seg_ids.max(axis=1).values.reshape((seg_ids.shape[0], 1))
    where_top = np.where(is_top)
    seg_ids.values[where_top] = 0  # do not change this to None

    pre_ids = seg_ids.loc[seg_ids[sources.id].max(axis=1) > 0].index.values
    post_ids = seg_ids.loc[seg_ids[targets.id].max(axis=1) > 0].index.values

    # Fetch pre- and postsynapses associated with these segments
    # It's cheaper to get them all in one go
    syn = query_connections(pre_ids, post_ids,
                            score_thresh=score_thresh,
                            db=None)

    # Now associate synapses with neurons
    seg2neuron = dict(zip(seg_ids.index.values,
                          seg_ids.columns[np.argmax(seg_ids.values, axis=1)]))

    syn['id_pre'] = syn.segmentid_pre.map(seg2neuron)
    syn['id_post'] = syn.segmentid_post.map(seg2neuron)

    # Let the clean-up BEGIN!

    # First drop nasty autapses
    if drop_autapses:
        syn = syn[syn.id_pre != syn.id_post]

    # Next drop synapses far away from our neurons
    if dist_thresh:
        syn['pre_close'] = False
        syn['post_close'] = False
        for id in np.unique(syn[['id_pre', 'id_post']].values.flatten()):
            neuron = unique_neurons.idx[id]
            tree = navis.neuron2KDTree(neuron)

            is_pre = syn.id_pre == id
            if np.any(is_pre):
                dist, ix = tree.query(syn.loc[is_pre, ['pre_x', 'pre_y', 'pre_z']].values,
                                      distance_upper_bound=dist_thresh)
                syn.loc[is_pre, 'pre_close'] = dist < float('inf')

            is_post = syn.id_post == id
            if np.any(is_post):
                dist, ix = tree.query(syn.loc[is_post, ['post_x', 'post_y', 'post_z']].values,
                                      distance_upper_bound=dist_thresh)
                syn.loc[is_post, 'post_close'] = dist < float('inf')

        # Drop connections where either pre- or postsynaptic site are too far
        # away from the neuron
        syn = syn[syn.pre_close & syn.post_close]

    # Drop duplicate connections, i.e. connections that connect the same pre-
    # and postsynaptic segmentation ID and are within a distance of 250nm
    dupl_thresh = 250
    if drop_duplicates:
        # We are dealing with this from a presynaptic perspective
        while True:
            pre_tree = cKDTree(syn[['pre_x', 'pre_y', 'pre_z']].values)
            pairs = pre_tree.query_pairs(r=dupl_thresh, output_type='ndarray')

            same_pre = syn.iloc[pairs[:, 0]].segmentid_pre.values == syn.iloc[pairs[:, 1]].segmentid_pre.values
            same_post = syn.iloc[pairs[:, 0]].segmentid_post.values == syn.iloc[pairs[:, 1]].segmentid_post.values
            same_cn = same_pre & same_post

            # If no more pairs to collapse break
            if same_cn.sum() == 0:
                break

            # Generate a graph from pairs
            G = nx.Graph()
            G.add_edges_from(pairs[same_cn])

            # Find the minimum number of nodes we need to remove
            # to separate the connectors
            to_rm = []
            for cn in nx.connected_components(G):
                to_rm += list(nx.minimum_node_cut(nx.subgraph(G, cn)))
            syn = syn.drop(index=syn.index.values[to_rm])

    if agglomerate:
        edges = syn.groupby(['id_pre', 'id_post'],
                            as_index=False).cleft_id.count()
        edges.rename({'cleft_id': 'weight'}, axis=1, inplace=True)
        edges.sort_values('weight', ascending=False, inplace=True)
        return edges.reset_index(drop=True)

    return syn
