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
import pymaid

import numpy as np
import pandas as pd
import trimesh as tm

from tqdm.auto import tqdm

from .. import utils, move, spine
use_pbars = utils.use_pbars

__all__ = ['segments_to_neuron', 'segments_to_skids', 'neuron_to_segments',
           'find_autoseg_fragments', 'find_fragments', 'find_missed_branches',
           'locs_to_segments']


def locs_to_segments(locs, mip=0, dataset='fafb-ffn1-20200412',
                     coordinates='voxel'):
    """Retrieve Google segmentation IDs at given location(s).

    Uses a service on hosted services.itanna.io by Eric Perlman and Davi Bock.

    Parameters
    ----------
    locs :          list-like
                    Array of x/y/z coordinates.
    mip :           int [0-9]
                    Scale to query. Lower mip = more precise but slower;
                    higher mip = faster but less precise (small segments
                    might not show at all at high mips).
    dataset :       str
                    Currently, the only available dataset is
                    "fafb-ffn1-20200412", the most recent segmentation by Google.
    coordinates :   "voxel" | "nm"
                    Units in which your coordinates are in. "voxel" is assumed
                    to be 4x4x40 (x/y/z) nanometers.

    Returns
    -------
    numpy.array
                List of segmentation IDs in the same order as ``locs``. Invalid
                locations will be returned with ID 0.

    """
    return spine.transform.get_segids(locs, segmentation=dataset,
                                      coordinates=coordinates, mip=mip)


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
    if isinstance(nl, navis.TreeNeuron):
        nl = pymaid.CatmaidNeuronList(nl)

    # Invert seg2skid
    skid2seg = {}
    for k, v in seg2skid.items():
        skid2seg[v] = skid2seg.get(v, []) + [k]

    for n in nl:
        n.seg_ids = skid2seg[int(n.id)]

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

    seg_ids = navis.utils.make_iterable(seg_ids)

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
    x :                 Neuron/List
                        Neurons for which to return segment IDs.

    Returns
    -------
    overlap_matrix :    pandas.DataFrame
                        DataFrame of segment IDs (rows) and IDs
                        (columns) with overlap in nodes as values::

                            skeleton_id  id  3245
                            seg_id
                            10336680915   5     0
                            10336682132   0     1

    """
    if isinstance(x, navis.TreeNeuron):
        x = navis.NeuronList(x)

    assert isinstance(x, navis.NeuronList)

    # We must not perform this on x.nodes as this is a temporary property
    nodes = x.nodes

    # Get segmentation IDs
    nodes['seg_id'] = locs_to_segments(nodes[['x', 'y', 'z']].values,
                                       coordinates='nm', mip=0)

    # Count segment IDs
    seg_counts = nodes.groupby(['neuron', 'seg_id'], as_index=False).node_id.count()
    seg_counts.columns = ['skeleton_id', 'seg_id', 'counts']

    # Remove seg IDs 0
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
    x :                 navis.Neuron/List
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
    >>> fafbseg.google.use_google_storage("https://storage.googleapis.com/fafb-ffn1-20190805/segmentation")

    Find autoseg fragments overlapping with a manually traced neuron:

    >>> x = pymaid.get_neuron(16, remote_instance=manual)
    >>> frags_of_x = fafbseg.google.find_autoseg_fragments(x, remote_instance=auto)

    See Also
    --------
    fafbseg.google.find_fragments
                        Generalization of this function that can find neurons
                        independent of whether they have a reference to the
                        segmentation ID in e.g. their name or annotations. Use
                        this to find manual tracings overlapping with a given
                        neuron e.g. from autoseg.

    """
    if not isinstance(x, navis.TreeNeuron):
        raise TypeError('Expected navis.TreeNeuron or pymaid.CatmaidNeuron, got "{}"'.format(type(x)))

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


def find_fragments(x, remote_instance, min_node_overlap=3, min_nodes=1,
                   mesh=None):
    """Find manual tracings overlapping with given autoseg neuron.

    This function is a generalization of ``find_autoseg_fragments`` and is
    designed to not require overlapping neurons to have references (e.g.
    in their name) to segmentation IDs:

        1. Traverse neurites of ``x`` search within 2.5 microns radius for
           potentially overlapping fragments.
        2. Either collect segmentation IDs for the input neuron and all
           potentially overlapping fragments or (if provided) use the mesh
           to check if the candidates are inside that mesh.
        3. Return fragments that overlap with at least ``min_overlap`` nodes
           with input neuron.

    Parameters
    ----------
    x :                 pymaid.CatmaidNeuron | navis.TreeNeuron
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
    mesh :              navis.Volume | trimesh.Trimesh | navis.MeshNeuron, optional
                        Mesh representation of ``x``. If provided will use the
                        mesh instead of querying the segmentation to determine
                        if fragments overlap. This is generally the faster.

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
    >>> fafbseg.google.use_google_storage("https://storage.googleapis.com/fafb-ffn1-20190805/segmentation")

    Find manually traced fragments overlapping with an autoseg neuron:

    >>> x = pymaid.get_neuron(204064470, remote_instance=auto)
    >>> frags_of_x = fafbseg.google.find_fragments(x, remote_instance=manual)

    See Also
    --------
    fafbseg.google.find_autoseg_fragments
                        Use this function if you are looking for autoseg
                        fragments overlapping with a given neuron. Because we
                        can use the reference to segment IDs (via names &
                        annotations), this function is much faster than
                        ``find_fragments``.

    """
    if not isinstance(x, navis.TreeNeuron):
        raise TypeError('Expected navis.TreeNeuron or pymaid.CatmaidNeuron, got "{}"'.format(type(x)))

    meshtypes = (type(None), navis.MeshNeuron, navis.Volume, tm.Trimesh)
    if not isinstance(mesh, meshtypes):
        raise TypeError(f'Unexpected mesh of type "{type(mesh)}"')

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
    tn_table = pymaid.get_node_table(skids,
                                     include_details=False,
                                     convert_ts=False,
                                     remote_instance=remote_instance)
    # Keep track of total node counts
    node_counts = tn_table.groupby('skeleton_id').node_id.count().to_dict()

    # If no mesh, use segmentation
    if not mesh:
        # Get segment IDs for the input neuron
        x.nodes['seg_id'] = locs_to_segments(x.nodes[['x', 'y', 'z']].values,
                                             coordinates='nm', mip=0)

        # Count segment IDs
        x_seg_counts = x.nodes.groupby('seg_id').node_id.count().reset_index(drop=False)
        x_seg_counts.columns = ['seg_id', 'counts']

        # Remove seg IDs 0
        x_seg_counts = x_seg_counts[x_seg_counts.seg_id != 0]

        # Generate KDTree for nearest neighbor calculations
        tree = navis.neuron2KDTree(x)

        # Now remove nodes that aren't even close to our input neuron
        dist, ix = tree.query(tn_table[['x', 'y', 'z']].values,
                              distance_upper_bound=2500)
        tn_table = tn_table.loc[dist <= 2500]

        # Remove neurons that can't possibly have enough overlap
        node_counts2 = tn_table.groupby('skeleton_id').node_id.count().to_dict()
        to_keep = [k for k, v in node_counts2.items() if v >= min(min_node_overlap, node_counts[k])]
        tn_table = tn_table[tn_table.skeleton_id.isin(to_keep)]

        # Add segment IDs
        tn_table['seg_id'] = locs_to_segments(tn_table[['x', 'y', 'z']].values,
                                              coordinates='nm', mip=0)

        # Now group by neuron and by segment
        seg_counts = tn_table.groupby(['skeleton_id', 'seg_id']).node_id.count().reset_index(drop=False)
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
    else:
        # Check if nodes are inside or outside the mesh
        tn_table['in_mesh'] = navis.in_volume(tn_table[['x', 'y', 'z']].values,
                                              mesh).astype(bool)
        # Count the number of in-mesh nodes for each neuron
        # This also drops skeletons without
        ol_counts = tn_table.groupby('skeleton_id', as_index=False).in_mesh.sum()

        # Rename columns
        ol_counts.columns = ['skeleton_id', 'counts']

        # Now subset to those that are overlapping sufficiently
        # First drop all non-overlapping fragments
        ol_counts = ol_counts[ol_counts.counts > 0]

        # Add column with total node counts
        ol_counts['node_count'] = ol_counts.skeleton_id.map(node_counts)

        # Generate an threshold array such that the threshold is the minimum
        # between the total nodes and min_node_overlap
        mno_array = np.repeat([min_node_overlap], ol_counts.shape[0])
        mnc_array = ol_counts.skeleton_id.map(node_counts).values
        thr_array = np.vstack([mno_array, mnc_array]).min(axis=0)

        # Subset to candidates that meet the threshold
        ol_counts = ol_counts[ol_counts.counts >= thr_array]

        ol = ol_counts.skeleton_id.tolist()
        scores = ol_counts.counts.tolist()

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


@utils.never_cache
def find_missed_branches(x, autoseg_instance, tag=False, tag_size_thresh=10,
                         min_node_overlap=4, **kwargs):
    """Use autoseg to find (and annotate) potential missed branches.

    Parameters
    ----------
    x :                 pymaid.CatmaidNeuron/List
                        Neuron(s) to search for missed branches.
    autoseg_instance :  pymaid.CatmaidInstance
                        CATMAID instance containing the autoseg skeletons.
    tag :               bool, optional
                        If True, will tag nodes of ``x`` that might have missed
                        branches with "missed branch?".
    tag_size_thresh :   int, optional
                        Size threshold in microns of cable for tagging
                        potentially missed branches.
    min_node_overlap :  int, optional
                        Minimum number of nodes that input neuron(s) x must
                        overlap with given segmentation ID for it to be
                        included.
    **kwargs
                        Keyword arguments passed to
                        ``fafbseg.neuron_from_segments``.

    Returns
    -------
    summary :           pandas.DataFrame
                        DataFrame containing a summary of potentially missed
                        branches.

                        If input is a single neuron:

    fragments :         pymaid.CatmaidNeuronList
                        Fragments found to be potentially overlapping with the
                        input neuron.
    branches :          pymaid.CatmaidNeuronList
                        Potentially missed branches extracted from ``fragments``.

    Examples
    --------
    Setup

    >>> import fafbseg
    >>> import pymaid

    >>> # Set up connections to manual and autoseg CATMAID
    >>> manual = pymaid.CatmaidInstance('URL', 'HTTP_USER', 'HTTP_PW', 'API_TOKEN')
    >>> auto = pymaid.CatmaidInstance('URL', 'HTTP_USER', 'HTTP_PW', 'API_TOKEN')

    >>> # Set a source for segmentation data
    >>> fafbseg.google.use_google_storage("https://storage.googleapis.com/fafb-ffn1-20190805/segmentation")

    Find missed branches and tag them

    >>> # Fetch a neuron
    >>> x = pymaid.get_neuron(16, remote_instance=manual)
    >>> # Find and tag missed branches
    >>> (summary,
    ...  fragments,
    ...  branches) = fafbseg.google.find_missed_branches(x, autoseg_instance=auto)

    >>> # Show summary of missed branches
    >>> summary.head()
       n_nodes  cable_length   node_id
    0      110     28.297424   3306395
    1       90     23.976504  20676047
    2       64     15.851333  23419997
    3       29      7.494350   6298769
    4       16      3.509739  15307841

    >>> # Co-visualize your neuron and potentially overlapping autoseg fragments
    >>> x.plot3d(color='w')
    >>> fragments.plot3d()

    >>> # Visualize the potentially missed branches
    >>> pymaid.clear3d()
    >>> x.plot3d(color='w')
    >>> branches.plot3d(color='r')

    """
    if isinstance(x, navis.NeuronList):
        to_concat = []
        for n in tqdm(x, desc='Processing neurons', disable=not use_pbars, leave=False):
            (summary,
             frags,
             branches) = find_missed_branches(n,
                                              autoseg_instance=autoseg_instance,
                                              tag=tag,
                                              tag_size_thresh=tag_size_thresh,
                                              **kwargs)
            summary['skeleton_id'] = n.id
            to_concat.append(summary)

        return pd.concat(to_concat, ignore_index=True)
    elif not isinstance(x, navis.TreeNeuron):
        raise TypeError(f'Input must be TreeNeuron/List, got "{type(x)}"')

    # Find autoseg neurons overlapping with input neuron
    nl = find_autoseg_fragments(x,
                                autoseg_instance=autoseg_instance,
                                min_node_overlap=min_node_overlap,
                                verbose=False,
                                raise_none_found=False)

    # Next create a union
    if not nl.empty:
        for n in nl:
            n.nodes['origin'] = 'autoseg'
            n.nodes['origin_skid'] = n.skeleton_id

        # Create a simple union
        union = navis.stitch_neurons(nl, method='NONE')

        # Merge into target neuron
        union, new_edges, clps_map = move.merge_utils.collapse_nodes(union, x, limit=2)

        # Subset to autoseg nodes
        autoseg_nodes = union.nodes[union.nodes.origin == 'autoseg'].node_id.values
    else:
        autoseg_nodes = np.empty((0, 5))

    # Process fragments if any autoseg nodes left
    data = []
    frags = navis.NeuronList([])
    if autoseg_nodes.shape[0]:
        autoseg = navis.subset_neuron(union, autoseg_nodes)

        # Split into fragments
        frags = navis.break_fragments(autoseg)

        # Generate summary
        nodes = union.nodes.set_index('node_id')
        for n in frags:
            # Find parent node in union
            pn = nodes.loc[n.root[0], 'parent_id']
            pn_co = nodes.loc[pn, ['x', 'y', 'z']].values
            org_skids = n.nodes.origin_skid.unique().tolist()
            data.append([n.n_nodes, n.cable_length, pn, pn_co, org_skids])

    df = pd.DataFrame(data, columns=['n_nodes', 'cable_length', 'node_id',
                                     'node_loc', 'autoseg_skids'])
    df.sort_values('cable_length', ascending=False, inplace=True)

    if tag and not df.empty:
        to_tag = df[df.cable_length >= tag_size_thresh].node_id.values

        resp = pymaid.add_tags(to_tag,
                               tags='missed branch?',
                               node_type='TREENODE',
                               remote_instance=x._remote_instance)

        if 'error' in resp:
            return df, resp

    return df, nl, frags
