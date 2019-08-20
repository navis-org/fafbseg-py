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

from . import utils
use_pbars = utils.use_pbars


def segments_to_neuron(seg_ids, autoseg_instance, name_pattern="Google: {id}",
                       verbose=True):
    """Retrieve autoseg neurons of given segmentation ID(s).

    If a given segmentation ID has been merged into another fragment, will try
    retrieving by annotation.

    Parameters
    ----------
    seg_ids :           int | list of int
                        Segmentation ID(s) of autoseg skeletons to retrieve.
    autoseg_instance :  pymaid.CatmaidInstance
                        Instance with autoseg skeletons.
    name_pattern :      str, optional
                        Segmentation IDs are encoded in the name. Use parameter
                        this to define that pattern.


    Returns
    -------
    CatmaidNeuronList

    """
    assert isinstance(autoseg_instance, pymaid.CatmaidInstance)

    seg_ids = pymaid.utils._make_iterable(seg_ids)

    # First find neurons by name
    names = [name_pattern.format(id=i) for i in seg_ids]
    by_name = pymaid.get_skids_by_name(names,
                                       allow_partial=False,
                                       remote_instance=autoseg_instance)

    to_fetch = by_name.skeleton_id.tolist()

    # Look for missing IDs
    missing = [n for n in names if n not in by_name.name.values]

    # Try finding by annotation (temporarily raise logger level)
    if missing:
        old_lvl = pymaid.logger.level
        pymaid.set_loggers('ERROR')
        by_annotation = pymaid.get_skids_by_annotation(missing,
                                                       raise_not_found=False,
                                                       remote_instance=autoseg_instance)
        pymaid.set_loggers(old_lvl)

        to_fetch += by_annotation

    if not to_fetch:
        raise ValueError("None of the provided segmentation IDs could be found")

    nl = pymaid.get_neurons(to_fetch,
                            remote_instance=autoseg_instance)

    # Figure out which we are still missing
    if verbose:
        nl.get_annotations()
        all_annotations = set([a for an in nl.annotations for a in an])

        missing = [s for n, s in zip(names, seg_ids) if n not in nl.neuron_name and n not in all_annotations]

        if missing:
            print("{} (of {}) segmentation ID(s) could not be found: {}".format(len(missing),
                                                                                len(seg_ids),
                                                                                ", ".join(missing)))

    return nl


def neuron_to_segments(x, **kwargs):
    """Use brainmaps API to return segment IDs overlapping with a given neuron.

    This is in essence a higher-level function of brainmappy's
    get_seg_at_location.

    Parameters
    ----------
    x :                 CatmaidNeuron/List
                        Neurons for which to return segment IDs.
    **kwargs
                        Keyword arguments passed to
                        `brainmappy.get_seg_at_location`. Use this to set a
                        lower `chunksize` (default 10e3) if you are
                        experiencing "Service Unavailable" errors.

    Returns
    -------
    overlap_matrix :    pandas.DataFrame
                        DataFrame of segment IDs (rows) and skeleton IDs
                        (columns) with overlap in nodes as values::

                            skeleton_id  16  3245
                            seg_id
                            10336680915   5     0
                            10336682132   0     1

    """
    if isinstance(x, pymaid.CatmaidNeuron):
        x = pymaid.CatmaidNeuronList(x)

    assert isinstance(x, pymaid.CatmaidNeuronList)

    # We must not perform this on x.nodes as this is a temporary property
    nodes = x.nodes

    # Get segmentation IDs
    nodes['seg_id'] = bm.get_seg_at_location(nodes[['x', 'y', 'z']].values,
                                             chunksize=kwargs.get('chunksize', 1e3))

    # Count segment IDs
    seg_counts = nodes.groupby(['skeleton_id', 'seg_id']).treenode_id.count().reset_index(drop=False)
    seg_counts.columns = ['skeleton_id', 'seg_id', 'counts']

    # Remove seg IDs 0 (glia?)
    seg_counts = seg_counts[seg_counts.seg_id != 0]

    # Turn into matrix where columns are skeleton IDs, segment IDs are rows
    # and values are the overlap counts
    matrix = seg_counts.pivot(index='seg_id', columns='skeleton_id', values='counts')

    return matrix


def find_fragments(x, remote_instance, min_node_overlap=3, min_nodes=1):
    """Find fragments constituting a neuron in another CatmaidInstance,
    e.g. manual tracings for an autoseg neuron.

    This function is designed to not require overlapping neurons to have
    references (e.g. in their name) to segmentation IDs:
        1. Traverse neurites of ``x`` search within 2.5 microns radius for
           potentially overlapping fragments.
        2. Collect segmentation IDs for the input neuron and all potentially
           overlapping fragments using the brainmaps API.
        3. Return fragments that overlap with at least ``min_overlap`` nodes
           with input neuron.


    Parameters
    ----------
    x :                 pymaid.CatmaidNeuron
                        Neuron to collect fragments for.
    remote_instance :   pymaid.CatmaidInstance
                        Catmaid instance in which to search for fragments.
    min_node_overlap :  int, optional
                        Minimal overlap between `x` and a fragment in nodes. If
                        the fragment has less total nodes than `min_overlap`,
                        the threshold will be lowered to:
                        ``min_overlap = min(min_overlap, fragment.n_nodes)
    min_nodes :         int, optional
                        Minimum node count for returned neurons.

    Return
    ------
    pymaid.CatmaidNeuronList
                        CatmaidNeurons of the overlapping fragments.
    list
                        List with number of nodes that each fragment overlaps
                        with input neuron.

    Examples
    --------
    >>> import pymaid
    >>> rm1 = pymaid.CatmaidInstance('SERVER_URL', 'HTTP_USER', 'HTTP_PW', 'API_TOKEN')
    >>> rm2 = pymaid.CatmaidInstance('SERVER_URL', 'HTTP_USER', 'HTTP_PW', 'API_TOKEN')

    >>> import brainmappy as bm
    >>> flow = bm.acquire_credentials('client_secret.json', make_global=True)
    >>> bm.set_global_volume('some_volume_id')

    >>> x = pymaid.get_neuron(204064470, remote_instance=rm1)
    >>> frags_of_x = find_fragments(x_auto, remote_instance=rm2)

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

    # Get nodes for these candidates
    tn_table = pymaid.get_treenode_table(skids,
                                         include_details=False,
                                         convert_ts=False,
                                         remote_instance=remote_instance)
    # Keep track of total node counts
    node_counts = tn_table.groupby('skeleton_id').treenode_id.count().to_dict()

    # Get segment IDs for the input neuron
    x.nodes['seg_id'] = bm.get_seg_at_location(x.nodes[['x', 'y', 'z']].values,
                                               chunksize=remote_instance.max_threads * 100)

    # Count segment IDs
    x_seg_counts = x.nodes.groupby('seg_id').treenode_id.count().reset_index(drop=False)
    x_seg_counts.columns = ['seg_id', 'counts']

    # Remove seg IDs 0
    x_seg_counts = x_seg_counts[x_seg_counts.seg_id != 0]

    # Generate KDTree for nearest neighbor calculations
    tree = pymaid.neuron2KDTree(x)

    # Now remove nodes that aren't even close to our input neuron
    dist, ix = tree.query(tn_table[['x', 'y', 'z']].values,
                          distance_upper_bound=2500)
    tn_table = tn_table.loc[dist <= 2500]

    # Add segment IDs
    tn_table['seg_id'] = bm.get_seg_at_location(tn_table[['x', 'y', 'z']].values,
                                                chunksize=remote_instance.max_threads * 100)

    # Now group by neuron and by segment
    seg_counts = tn_table.groupby(['skeleton_id', 'seg_id']).treenode_id.count().reset_index(drop=False)
    # Rename columns
    seg_counts.columns = ['skeleton_id', 'seg_id', 'counts']

    # Remove seg IDs 0
    seg_counts = seg_counts[seg_counts.seg_id != 0]

    # Remove segments IDs that are not overlapping with input neuron
    seg_counts = seg_counts[np.isin(seg_counts.seg_id.values,
                                    x_seg_counts.seg_id.values)]

    # Now go over each candidate and see if there is enough overlap
    ol = []
    scores = []
    for s in seg_counts.skeleton_id.unique():
        # Subset to nodes of this neurons
        this_counts = seg_counts[seg_counts.skeleton_id == s]

        # If the neuron is smaller than min_overlap, lower the threshold
        this_min_ol = min(min_node_overlap, node_counts[s])

        # Sum up counts for both input neuron and this candidate
        c_count = this_counts.counts.sum()
        x_count = x_seg_counts[x_seg_counts.seg_id.isin(this_counts.seg_id.values)].counts.sum()

        # If there is enough overlap, keep this neuron
        # and add score as `overlap_score`
        if (c_count >= this_min_ol) and (x_count >= this_min_ol):
            # The score is the minimal overlap
            scores.append(min(c_count, x_count))
            ol.append(s)

    if ol:
        ol = pymaid.get_neurons(ol, remote_instance=remote_instance)

        # Make sure it's a neuronlist
        if not isinstance(ol, pymaid.CatmaidNeuronList):
            ol = pymaid.CatmaidNeuronList(ol)

        for n, s in zip(ol, scores):
            n.overlap_score = s
    else:
        ol = pymaid.CatmaidNeuronList([])

    return ol
