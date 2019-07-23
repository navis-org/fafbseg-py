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

import brainmappy as bm
import numpy as np
import pandas as pd
import pymaid

from pymaid.cache import never_cache

from tqdm import tqdm


@never_cache
def find_fragments(x, remote_instance, min_overlap=3, min_nodes=1):
    """Find fragments constituting a neuron in another CatmaidInstance,
    e.g. manual tracings for an autoseg neuron.

    This function works by:
        1. Traverse neurites of ``x`` and find neurons within 2.5 microns radius.
        2. Collect segmentation IDs for the input neuron and all potentially
           overlapping fragments using the brainmaps API.
        3. Return fragments that overlap with at least ``min_overlap`` nodes
           in the same segmentation ID with the input neuron.


    Parameters
    ----------
    x :                 pymaid.CatmaidNeuron
                        Neuron to collect fragments for.
    remote_instance :   pymaid.CatmaidInstance
                        Catmaid instance in which to search for fragments.
    min_overlap :       int, optional
                        Minimal overlap between `x` and a fragment. If the
                        fragment has less nodes than `min_overlap`, the
                        threshold will be lowered accordingly.
    min_nodes :         int, optional
                        Minimum node count for returned neurons. For each
                        fragment the threshold is::

                            min(min_nodes, total nodes of fragment)

    Return
    ------
    pymaid.CatmaidNeuronList
                        CatmaidNeurons of the overlapping fragments.

    Examples
    --------
    >>> import pymaid
    >>> auto = pymaid.CatmaidInstance('http://your-server/auto', 'http_user', 'http_pw', 'api_token')
    >>> manual = pymaid.CatmaidInstance('http://your-server/manual', 'http_user', 'http_pw', 'api_token')

    >>> import brainmappy as bm
    >>> flow = bm.acquire_credentials('client_secret.json', make_global=True)
    >>> bm.set_global_volume('some_volume_id')

    >>> x_auto = pymaid.get_neuron(204064470, remote_instance=auto)
    >>> x_man = find_fragments(x_auto, remote_instance=manual)

    """
    if not isinstance(x, pymaid.CatmaidNeuron):
        raise TypeError('Expected pymaid.CatmaidNeuron, got "{}"'.format(type(x)))

    # Resample the autoseg neuron to 0.5 microns
    x_rs = x.resample(500, inplace=False)

    # For each node get skeleton IDs in a 0.25 micron radius
    r = 250
    # Generate bounding boxes around each node
    bboxes = [np.vstack([co - r, co + r]).T for co in x_rs.nodes[['x', 'y', 'z']].values]

    # Query each bounding box
    urls = [remote_instance._get_skeletons_in_bbox(minx=min(b[0]),
                                                   maxx=max(b[0]),
                                                   miny=min(b[1]),
                                                   maxy=max(b[1]),
                                                   minz=min(b[2]),
                                                   maxz=max(b[2]),
                                                   min_nodes=min_nodes) for b in bboxes]
    resp = remote_instance.fetch(urls, desc='Searching for overlapping neurons')
    skids = set([s for l in resp for s in l])

    # Get these candidates
    cand = pymaid.get_neurons(skids, remote_instance=remote_instance)

    # Get segment IDs for the input neuron
    x_segs = bm.get_seg_at_location(x.nodes[['x', 'y', 'z']].values)

    # Remove segment ID "0" which is glia (I believe) and count occurrences
    x_segs = np.array(x_segs)
    x_uni, x_count = np.unique(x_segs[x_segs != "0"], return_counts=True)

    # Discard segs with lower than min_overlap counts
    x_uni = x_uni[x_count >= min_overlap]

    tree = pymaid.neuron2KDTree(x)

    # Go over each candidate and collect positions to query
    nodes = []
    for c in cand:
        # Find points in this neuron that are actually close to the input neuron
        dist, ix = tree.query(c.nodes[['x', 'y', 'z']].values,
                              distance_upper_bound=2500)

        # If the neuron is smaller than min_overlap, lower the threshold
        this_min_ol = min(min_overlap, c.nodes.shape[0])

        # Skip prematurely if there is no way for us to get enough overlap
        if sum(dist <= 2500) < this_min_ol:
            continue

        this_nodes = c.nodes.loc[dist <= 2500,
                                 ['treenode_id', 'x', 'y', 'z']]
        this_nodes['skeleton_id'] = c.skeleton_id

        nodes.append(this_nodes)

    # Turn this into a single stack
    nodes = pd.concat(nodes, axis=0)

    # Add segment IDs
    nodes['seg_id'] = bm.get_seg_at_location(nodes[['x', 'y', 'z']].values)

    # Go over each candidate and if there is enough overlap
    ol = []
    for c in cand:
        # Subset to nodes that we queried
        this_nodes = nodes[nodes.skeleton_id == c.skeleton_id]

        # Skip prematurely if there are no nodes to check
        if this_nodes.empty:
            continue

        # If the neuron is smaller than min_overlap, lower the threshold
        this_min_ol = min(min_overlap, c.nodes.shape[0])

        # Count unique segments (discarding "0" which is glia)
        c_seg = np.array(this_nodes['seg_id'].values)
        c_uni, c_count = np.unique(c_seg[c_seg != "0"], return_counts=True)

        # Find segment IDs that occur more than min_overlap and are also present
        # in input neuron
        c_ol = (c_count >= this_min_ol) & np.isin(c_uni, x_uni)
        # NOTE: this is currently filtering to min_overlap per segment
        #       -> we should make this per fragment

        # If there is overlap, keep this neuron
        if any(c_ol):
            ol.append(c)

    return pymaid.CatmaidNeuronList(ol)
