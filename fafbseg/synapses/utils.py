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

"""Utility functions to work with synapse data."""

import networkx as nx
import numpy as np
import pandas as pd

from scipy.spatial import cKDTree
from tqdm.auto import tqdm

__all__ = ['assign_connectors', 'process_synapse_table', 'catmaid_table']


def catmaid_table(cn_table, query_ids):
    """Style connectivity table like in CATMAID.

    Parameters
    ----------
    cn_table :  pd.DataFrame
                Plain synapse table. Must contain "pre", "post" and "weight"
                columns.
    query_ids : list of int
                IDs of the neurons originally queried for this table.

    Returns
    -------
    pd.DataFrame
                CATMAID style partner table.

    """
    assert isinstance(cn_table, pd.DataFrame)
    assert all([c in cn_table.columns for c in ['pre', 'post', 'weight']])

    up = cn_table[cn_table.post.isin(query_ids)]
    dn = cn_table[cn_table.pre.isin(query_ids)]

    up = up.pivot(index='pre', columns='post', values='weight').reindex(query_ids, axis=1).fillna(0)
    up['relation'] = 'upstream'
    dn = dn.pivot(index='post', columns='pre', values='weight').reindex(query_ids, axis=1).fillna(0)
    dn['relation'] = 'downstream'
    dn.index.name = up.index.name = 'id'
    dn.columns.name = up.columns.name = None

    cmb = pd.concat((up, dn), axis=0, join='outer', sort=True)
    cmb = cmb[['relation'] + list(query_ids)]
    cmb['total'] = cmb[query_ids].sum(axis=1)
    cmb.sort_values(['relation', 'total'], inplace=True, ascending=False)

    return cmb


def process_synapse_table(syn,  min_score=30, dist_thresh=1000, drop_autapses=True,
                          drop_duplicates=True, collapse_connectors=False,
                          verbose=True, progress=True):
    """Clean and reformat synapse table.

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
    syn :           pd.DataFrame
                    The synapse table to clean and reformat.
    min_score :     int, optional
                    If provided will drop synapses with a cleft score lower than
                    this.
    dist_thresh :   int
                    Synapses further away than the given distance [nm] from any
                    node in the neuron than given distance will be discarded.
                    This is always with respect to the cleft site
                    irrespective of whether our neuron is pre- or postsynaptic
                    to it.
    drop_autapses : bool
                    If True, will drop synapses where pre- and postsynapse point
                    to the same neuron. Autapses are wrong in 99.9% of all cases
                    we've seen.
    drop_duplicates :   bool
                        If True, will merge synapses which connect the same
                        pair of pre-post segmentation IDs and are within
                        less than 2500nm.
    collapse_connectors : bool
                    If True, we will pool presynaptic connections into
                    CATMAID-like connectors. If False each row represents a
                    connections.
    progress :      bool
                    Whether to show progress bars or not.

    Return
    ------
    pd.DataFrame
                    Cleaned and reformatted synapse table.

    """
    if min_score:
        # Drop too low cleft cores
        syn = syn[syn.cleft_scores >= min_score].copy()

    # Drop autapses - they are most likely wrong
    if drop_autapses:
        syn = syn[syn.pre != syn.post]

    # Drop duplicates
    if drop_duplicates:
        dupl_thresh = 250
        # This is done iteratively
        while True:
            # Generate pairs of pre- and postsynaptic coordinates that are
            # suspiciously close
            pre_tree = cKDTree(syn[['pre_x', 'pre_y', 'pre_z']].values)
            post_tree = cKDTree(syn[['post_x', 'post_y', 'post_z']].values)
            pre_pairs = pre_tree.query_pairs(r=dupl_thresh)
            post_pairs = post_tree.query_pairs(r=dupl_thresh)

            # We consider pairs for removal where pre- OR postsynapse are close
            # Alternatively, we could also look for cases where both pre- AND
            # postsynapse must be too close for comfort
            # This behaviour is easy to change via below operator
            pairs = pre_pairs | post_pairs  # union of both
            pairs = np.array(list(pairs))

            # For each pair check if they connect the same IDs
            same_pre = syn.iloc[pairs[:, 0]].pre.values == syn.iloc[pairs[:, 1]].pre.values
            same_post = syn.iloc[pairs[:, 0]].post.values == syn.iloc[pairs[:, 1]].post.values
            same_cn = same_pre & same_post

            # If no more pairs to collapse break loop
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
        # Reset index so it's continuous again
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

        connectors['neuron'] = neuron.id  # do NOT change the type of this
        tables.append(connectors)

    connectors = pd.concat(tables, axis=0, sort=True).reset_index(drop=True)
    return connectors



def assign_connectors(synapses, max_dist=300):
    """Collapse synapses by presynaptic connectors.

    1. Iterate over each unique presynaptic segment ID and ...
    2. Form pairs of presynapses that are within ``max_dist``
    3. Use these pairs to create a network graph
    4. Break graph into connected components
    3. Give a unique connector ID to all synapses in a connected component

    Parameters
    ----------
    synapse :   pandas.DataFrame
                Must contains ['pre_x', 'pre_y', 'pre_z'] or column.
    max_dist :  int
                Max distance at which two presynaptic locations may be
                collapsed.

    Returns
    -------
    None
                Adds a `connector_id` column to DataFrame.

    """
    assert isinstance(synapses, pd.DataFrame)

    if all(np.isin(['pre_x', 'pre_y', 'pre_z'], synapses.columns)):
        loc_cols = ['pre_x', 'pre_y', 'pre_z']
    elif all(np.isin(['x', 'y', 'z'], synapses.columns)):
        loc_cols = ['x', 'y', 'z']
    else:
        raise ValueError('Need either pre_x/pre_y/pre_z or x/y/z columns.')

    # Iterate over presynaptic IDs
    cn_id_counter = 1
    synapses['connector_id'] = None
    for sid in tqdm(synapses.segmentid_pre.unique(),
                    leave=False,
                    desc='Collapsing synapses'):
        this_sid = synapses.segmentid_pre == sid
        this = synapses[this_sid]

        if this.shape[0] == 1:
            synapses.loc[this_sid, 'connector_id'] = cn_id_counter
            cn_id_counter += 1
            continue

        # Build KDTree and generate pairs
        this_locs = this[loc_cols].values
        tree = cKDTree(this_locs)
        pairs = tree.query_pairs(r=max_dist)

        # Generate graph from pairs
        G = nx.Graph()
        G.add_nodes_from(np.arange(this.shape[0]))  # this is for lone synapses
        G.add_edges_from(pairs)

        this_cn_ids = np.zeros(this.shape[0])
        for cc in nx.connected_components(G):
            this_cn_ids[list(cc)] = cn_id_counter
            cn_id_counter += 1

        synapses.loc[this_sid, 'connector_id'] = this_cn_ids

    synapses['connector_id'] = synapses.connector_id.astype(np.int32)
