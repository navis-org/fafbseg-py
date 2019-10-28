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

import numpy as np
import pandas as pd
import pymaid

from . import utils, segmentation
use_pbars = utils.use_pbars


def segments_to_neuron(seg_ids, autoseg_instance, name_pattern="Google: {id}",
                       verbose=True, raise_none_found=True):
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
    raise_none_found :  bool, optional
                        If True and none of the requested segments were found,
                        will raise ValueError

    Returns
    -------
    CatmaidNeuronList
                        Neurons representing given segmentation IDs. Each neuron
                        has the list of segmentation IDs associated with it
                        as ``.seg_ids`` attribute.

    """
    assert isinstance(autoseg_instance, pymaid.CatmaidInstance)

    seg2skid = segments_to_skids(seg_ids,
                                 autoseg_instance=autoseg_instance,
                                 verbose=verbose)

    to_fetch = list(set([v for v in seg2skid.values() if v]))

    if not to_fetch:
        if raise_none_found:
            raise ValueError("None of the provided segmentation IDs could be found")
        else:
            # Return empty list
            return pymaid.CatmaidNeuronList([])

    nl = pymaid.get_neurons(to_fetch,
                            remote_instance=autoseg_instance)

    # Make sure we're dealing with a list of neurons
    if isinstance(nl, pymaid.CatmaidNeuron):
        nl = pymaid.CatmaidNeuronList(nl)

    # Invert seg2skid
    skid2seg = {}
    for k, v in seg2skid.items():
        skid2seg[v] = skid2seg.get(v, []) + [k]

    for n in nl:
        n.seg_ids = skid2seg[int(n.skeleton_id)]

    return nl


def segments_to_skids(seg_ids, autoseg_instance, name_pattern="Google: {id}",
                      merge_annotation_pattern="Merged: {name}",
                      verbose=True):
    """Retrieve skeleton IDs of neurons corresponding to given segmentation ID(s).

    If a given segmentation ID has been merged into another fragment, will try
    retrieving by annotation.

    Parameters
    ----------
    seg_ids :                   int | list of int
                                Segmentation ID(s) of autoseg skeletons to retrieve.
    autoseg_instance :          pymaid.CatmaidInstance
                                Instance with autoseg skeletons.
    name_pattern :              str, optional
                                Segmentation IDs are encoded in the name. Use
                                this parameter to define that pattern.
    merge_annotation_pattern :  str, optional
                                When neurons are merged, a reference to the
                                loosing skeleton's name is kept as annotation.
                                Use this parameter to define that pattern.

    Returns
    -------
    Dict
                        Dictionary mapping segmentation ID to skeleton ID.
                        Will be ``None`` if no skeleton found.

    """
    assert isinstance(autoseg_instance, pymaid.CatmaidInstance)

    assert isinstance(seg_ids, (list, np.ndarray, set, tuple, pd.Index, int, str))

    seg_ids = pymaid.utils._make_iterable(seg_ids)

    # Prepare map seg ID -> skeleton ID
    seg2skid = {int(i): None for i in seg_ids}

    # First find neurons by name
    # Do NOT change the order of "names"!
    names = [name_pattern.format(id=i) for i in seg_ids]
    by_name = pymaid.get_skids_by_name(names,
                                       allow_partial=False,
                                       raise_not_found=False,
                                       remote_instance=autoseg_instance)
    by_name['skeleton_id'] = by_name.skeleton_id.astype(int)

    # Update map by those that could be found by name
    name2skid = by_name.set_index('name').skeleton_id.to_dict()
    seg2skid.update({int(i): int(name2skid[n]) for i, n in zip(seg_ids, names) if n in by_name.name.values})

    # Look for missing IDs
    not_found = [s for s in seg_ids if not seg2skid[int(s)]]

    # Try finding by annotation (temporarily raise logger level)
    if not_found:
        map = merge_annotation_pattern
        an = [map.format(name=name_pattern.format(id=n)) for n in not_found]
        old_lvl = pymaid.logger.level
        pymaid.set_loggers('ERROR')
        by_annotation = pymaid.get_skids_by_annotation(an,
                                                       raise_not_found=False,
                                                       allow_partial=False,
                                                       intersect=False,
                                                       remote_instance=autoseg_instance)
        pymaid.set_loggers(old_lvl)

        if by_annotation:
            annotations = pymaid.get_annotations(by_annotation,
                                                 remote_instance=autoseg_instance)

            for seg, a in zip(not_found, an):
                for skid in annotations:
                    if a in annotations[skid]:
                        seg2skid[int(seg)] = int(skid)
                        break

    # Figure out if we are still missing skeletons for any of the seg IDs
    if verbose:
        missing = [str(k) for k, v in seg2skid.items() if not v]
        if missing:
            # Check if skeleton ID has ever existed
            hist = pymaid.get_skeleton_change(missing,
                                              remote_instance=autoseg_instance)
            # Flatten the list of links (and convert to string)
            existed = set([str(e) for l in hist for e in l[0]])

            still_missing = set(missing) & existed

            if still_missing:
                msg = "{} out of {} segmentation IDs could not be found: {}"
                msg = msg.format(len(still_missing), len(seg_ids), ", ".join(still_missing))
                print(msg)

    return seg2skid


def neuron_to_segments(x):
    """Get segment IDs overlapping with a given neuron.

    Parameters
    ----------
    x :                 CatmaidNeuron/List
                        Neurons for which to return segment IDs.

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
    nodes['seg_id'] = segmentation.get_seg_ids(nodes[['x', 'y', 'z']].values)

    # Count segment IDs
    seg_counts = nodes.groupby(['skeleton_id', 'seg_id']).treenode_id.count().reset_index(drop=False)
    seg_counts.columns = ['skeleton_id', 'seg_id', 'counts']

    # Remove seg IDs 0 (glia?)
    seg_counts = seg_counts[seg_counts.seg_id != 0]

    # Turn into matrix where columns are skeleton IDs, segment IDs are rows
    # and values are the overlap counts
    matrix = seg_counts.pivot(index='seg_id', columns='skeleton_id', values='counts')

    return matrix


def find_autoseg_fragments(x, autoseg_instance, min_node_overlap=3, min_nodes=1,
                           verbose=True, raise_none_found=True):
    """Find autoseg tracings constituting a given neuron.

    This function works by querying the segmentation IDs along the neurites of
    your query neuron and then fetching the corresponding skeletons in
    ``autoseg_instance``.

    Parameters
    ----------
    x :                 pymaid.CatmaidNeuron
                        Neuron to collect fragments for.
    autoseg_instance :  pymaid.CatmaidInstance
                        Catmaid instance which contains autoseg fragments.
    min_node_overlap :  int, optional
                        Minimal overlap between `x` and a segmentation
                        [in nodes].
    min_nodes :         int, optional
                        Minimum node count for returned neurons.
    verbose :           bool, optional
                        If True, will be print summary of who has contributed
                        to the autoseg fragments by merging or tracing nodes.
    raise_none_found :  bool, optional
                        If True and none of the requested segments were found,
                        will raise ValueError

    Return
    ------
    pymaid.CatmaidNeuronList
                        CatmaidNeurons of the overlapping fragments.

    Examples
    --------
    Setup:

    >>> import pymaid
    >>> import fafbseg

    >>> manual = pymaid.CatmaidInstance('MANUAL_SERVER_URL', 'HTTP_USER', 'HTTP_PW', 'API_TOKEN')
    >>> auto = pymaid.CatmaidInstance('AUTO_SERVER_URL', 'HTTP_USER', 'HTTP_PW', 'API_TOKEN')

    >>> # Call one of the fafbseg.use_... to set a source for segmentation IDs
    >>> fafbseg.use_google_storage("https://storage.googleapis.com/fafb-ffn1-20190805/segmentation")

    Find autoseg fragments overlapping with a manually traced neuron:

    >>> x = pymaid.get_neuron(16, remote_instance=manual)
    >>> frags_of_x = fafbseg.find_autoseg_fragments(x, remote_instance=auto)

    See Also
    --------
    fafbseg.find_fragments
                        Generalization of this function that can find neurons
                        independent of whether they have a reference to the
                        segmentation ID in e.g. their name or annotations. Use
                        this to find manual tracings overlapping with a given
                        neuron e.g. from autoseg.

    """
    if not isinstance(x, pymaid.CatmaidNeuron):
        raise TypeError('Expected pymaid.CatmaidNeuron, got "{}"'.format(type(x)))

    if not isinstance(autoseg_instance, pymaid.CatmaidInstance):
        raise TypeError('Expected pymaid.CatmaidInstance, got "{}"'.format(type(autoseg_instance)))

    # First collect segments constituting this neuron
    overlap_matrix = neuron_to_segments(x)

    # If none found, return empty neuronlist
    if overlap_matrix.empty:
        return pymaid.CatmaidNeuronList([])

    # Filter
    seg_ids = overlap_matrix.loc[overlap_matrix[x.skeleton_id] >= min_node_overlap].index.tolist()

    # Now try finding the corresponding skeletons
    nl = segments_to_neuron(seg_ids,
                            autoseg_instance=autoseg_instance,
                            verbose=verbose,
                            raise_none_found=raise_none_found)

    nl.sort_values('n_nodes')

    # Give contribution summary
    if verbose:
        # Get annotation details
        an_details = pymaid.get_annotation_details(nl,
                                                   remote_instance=autoseg_instance)
        # Extract merge annotations
        merge_an = an_details[an_details.annotation.str.contains('Merged:')]
        # Group by user and count
        merge_count = merge_an.groupby('user')[['annotation']].count()

        # Get manual tracing contributions
        contr = pymaid.get_user_contributions(nl,
                                              remote_instance=autoseg_instance)
        contr.set_index('user', inplace=True)

        # Merge both
        summary = pd.merge(merge_count, contr[['nodes', 'nodes_reviewed']],
                           left_index=True, right_index=True).fillna(0)
        summary.columns = ['merges', 'nodes_traced', 'nodes_reviewed']

        if not summary.empty:
            print('Tracer have worked on these neurons:')
            print(summary)
        else:
            print('Nobody has worked on these neurons')

    return nl[nl.n_nodes >= min_nodes]


def find_fragments(x, remote_instance, min_node_overlap=3, min_nodes=1):
    """Find manual tracings overlapping with given autoseg neuron.

    This function is a generalization of ``find_autoseg_fragments`` and is
    designed to not require overlapping neurons to have references (e.g.
    in their name) to segmentation IDs:

        1. Traverse neurites of ``x`` search within 2.5 microns radius for
           potentially overlapping fragments.
        2. Collect segmentation IDs for the input neuron and all potentially
           overlapping fragments using whatever source for segmentation IDs
           you have set by ``fafbseg.use_...``.
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
                        ``min_overlap = min(min_overlap, fragment.n_nodes)``
    min_nodes :         int, optional
                        Minimum node count for returned neurons.

    Return
    ------
    pymaid.CatmaidNeuronList
                        CatmaidNeurons of the overlapping fragments. Overlap
                        scores are attached to each neuron as ``.overlap_score``
                        attribute.

    Examples
    --------
    Setup:

    >>> import pymaid
    >>> import fafbseg

    >>> manual = pymaid.CatmaidInstance('MANUAL_SERVER_URL', 'HTTP_USER', 'HTTP_PW', 'API_TOKEN')
    >>> auto = pymaid.CatmaidInstance('AUTO_SERVER_URL', 'HTTP_USER', 'HTTP_PW', 'API_TOKEN')

    >>> # Set a source for segmentation data
    >>> fafbseg.use_google_storage("https://storage.googleapis.com/fafb-ffn1-20190805/segmentation")

    Find manually traced fragments overlapping with an autoseg neuron:

    >>> x = pymaid.get_neuron(204064470, remote_instance=auto)
    >>> frags_of_x = fafbseg.find_fragments(x, remote_instance=manual)

    See Also
    --------
    fafbseg.find_autoseg_fragments
                        Use this function if you are looking for autoseg
                        fragments overlapping with a given neuron. Because we
                        can use the reference to segment IDs (via names &
                        annotations), this function is much faster than
                        ``find_fragments``.

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

    # Return empty NeuronList if no skids found
    if not skids:
        return pymaid.CatmaidNeuronList([])

    # Get nodes for these candidates
    tn_table = pymaid.get_treenode_table(skids,
                                         include_details=False,
                                         convert_ts=False,
                                         remote_instance=remote_instance)
    # Keep track of total node counts
    node_counts = tn_table.groupby('skeleton_id').treenode_id.count().to_dict()

    # Get segment IDs for the input neuron
    x.nodes['seg_id'] = segmentation.get_seg_ids(x.nodes[['x', 'y', 'z']].values)

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

    # Remove neurons that can't possibly have enough overlap
    node_counts2 = tn_table.groupby('skeleton_id').treenode_id.count().to_dict()
    to_keep = [k for k, v in node_counts2.items() if v >= min(min_node_overlap, node_counts[k])]
    tn_table = tn_table[tn_table.skeleton_id.isin(to_keep)]

    # Add segment IDs
    tn_table['seg_id'] = segmentation.get_seg_ids(tn_table[['x', 'y', 'z']].values)

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
