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

import numpy as np
import pymaid
import navis
import random

from tqdm import tqdm

from .. import utils
from ..google import find_fragments

from .merge_utils import collapse_nodes
from .interfaces import confirm_overlap

import inquirer
from inquirer.themes import GreenPassion

# This is to prevent FutureWarning from numpy (via vispy)
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

use_pbars = utils.use_pbars

__all__ = ['merge_into_catmaid']


@utils.never_cache
def merge_into_catmaid(x, target_instance, tag, min_node_overlap=4, min_overlap_size=1,
                       merge_limit=1, min_upload_size=0, min_upload_nodes=1,
                       update_radii=True, import_tags=False, label_joins=True,
                       sid_from_nodes=True, mesh=None):
    """Merge neuron into target CATMAID instance.

    This function will attempt to:

        1. Find fragments in ``target_instance`` that overlap with ``x``
           using whatever segmentation data source you have set using
           ``fafbseg.use_...``.
        2. Generate a union of these fragments and ``x``.
        3. Make a differential upload of the union leaving existing nodes
           untouched.
        4. Join uploaded and existing tracings into a single continuous
           neuron. This will also upload connectors but no node tags.

    Disclaimer:

     As with all imports to CATMAID, the importing user is responsible for
     the quality of the imported skeleton and to make sure no existing
     tracings (including annotations) are negatively impacted. Also note that
     import requires special CATMAID user permissions.

    Parameters
    ----------
    x :                 pymaid.CatmaidNeuron/List | navis.TreeNeuron/List
                        Neuron(s)/fragment(s) to commit to ``target_instance``.
    target_instance :   pymaid.CatmaidInstance
                        Target Catmaid instance to commit the neuron to.
    tag :               str
                        A tag to be added as part of a ``{URL} upload {tag}``
                        annotation. This should be something identifying your
                        group - e.g. ``tag='WTCam'`` for the Cambridge Wellcome
                        Trust group.
    min_node_overlap :  int, optional
                        Minimal overlap between `x` and a potentially
                        overlapping neuron in ``target_instance``. If
                        the fragment has less total nodes than `min_overlap`,
                        the threshold will be lowered to:
                        ``min_overlap = min(min_overlap, fragment.n_nodes)``
    min_overlap_size :  int, optional
                        Minimum node count for potentially overlapping neurons
                        in ``target_instance``. Use this to e.g. exclude
                        single-node synapse orphans.
    merge_limit :       int, optional
                        Distance threshold [um] for collapsing nodes of ``x``
                        into overlapping fragments in target instance. Decreasing
                        this will help if your neuron has complicated branching
                        patterns (e.g. uPN dendrites) at the cost of potentially
                        creating duplicate parallel tracings in the neuron's
                        backbone.
    min_upload_size :   float, optional
                        Minimum size in microns for upload of new branches:
                        branches found in ``x`` but not in the overlapping
                        neuron(s) in ``target_instance`` are uploaded in
                        fragments. Use this parameter to exclude small branches
                        that might not be worth the additional review time.
    min_upload_nodes :  int, optional
                        As ``min_upload_size`` but for number of nodes instead
                        of cable length.
    update_radii :      bool, optional
                        If True, will use radii in ``x`` to update radii of
                        overlapping fragments if (and only if) the nodes
                        do not currently have a radius (i.e. radius<=0).
    import_tags :       bool, optional
                        If True, will import node tags. Please note that this
                        will NOT import tags of nodes that have been collapsed
                        into manual tracings.
    label_joins :       bool, optional
                        If True, will label nodes at which old and new
                        tracings have been joined with tags ("Joined from ..."
                        and "Joined with ...") and with a lower confidence of
                        1.
    sid_from_nodes :    bool, optional
                        If True and the to-be-merged neuron has a "skeleton_id"
                        column it will be used to set the ``source_id`` upon
                        uploading new branches. This is relevant if your neuron
                        is a virtual chimera of several neurons: in order to
                        preserve provenance (i.e. correctly associating each
                        node with a ``source_id`` origin).
    mesh :              Volume | MeshNeuron | mesh-like object | list thereof
                        Mesh representation of ``x``. If provided, will use to
                        improve merging. If ``x`` is a list of neurons, must
                        provide a mesh for each of them.

    Returns
    -------
    Nothing
                        If all went well.
    dict
                        If something failed, returns server responses with
                        error logs.

    Examples
    --------
    Setup

    >>> import fafbseg
    >>> import pymaid

    >>> # Set up connections to manual and autoseg CATMAID
    >>> manual = pymaid.CatmaidInstance('URL', 'HTTP_USER', 'HTTP_PW', 'API_TOKEN')
    >>> auto = pymaid.CatmaidInstance('URL', 'HTTP_USER', 'HTTP_PW', 'API_TOKEN')

    >>> # Set a segmentation data source
    >>> fafbseg.google.use_google_storage("https://storage.googleapis.com/fafb-ffn1-20190805/segmentation")

    Merge a neuron from autoseg into v14

    >>> # Fetch the autoseg neuron to transfer to v14
    >>> x = pymaid.get_neuron(267355161, remote_instance=auto)

    >>> # Get the neuron's annotations so that they can be merged too
    >>> x.get_annotations(remote_instance=auto)

    >>> # Start the commit
    >>> # See online documentation for video of merge process
    >>> resp = fafbseg.move.merge_neuron(x, target_instance=manual)

    """
    if not isinstance(x, navis.NeuronList):
        if not isinstance(x, navis.TreeNeuron):
            raise TypeError('Expected TreeNeuron/List, got "{}"'.format(type(x)))
        x = navis.NeuronList(x)

    if not isinstance(mesh, (np.ndarray, list)):
        if isinstance(mesh, type(None)):
            mesh = [mesh] * len(x)
        else:
            mesh = [mesh]

    if len(mesh) != len(x):
        raise ValueError(f'Got {len(mesh)} meshes for {len(x)} neurons.')

    # Make a copy - in case we make any changes to the neurons
    # (like changing duplicate skeleton IDs)
    x = x.copy()

    if not isinstance(tag, (str, type(None))):
        raise TypeError('Tag must be string, got "{}"'.format(type(tag)))

    # Check user permissions
    perm = target_instance.fetch(target_instance.make_url('permissions'))
    requ_perm = ['can_annotate', 'can_annotate_with_token', 'can_import']
    miss_perm = [p for p in requ_perm if
                 target_instance.project_id not in perm[0].get(p, [])]

    if miss_perm:
        msg = 'You lack permissions: {}. Please contact an administrator.'
        raise PermissionError(msg.format(', '.join(miss_perm)))

    pymaid.set_loggers('WARNING')

    # Throttle requests just to play it safe
    # On a bad connection one might have to decrease max_threads further
    target_instance.max_threads = min(target_instance.max_threads, 50)

    # For user convenience, we will do all the stuff that needs user
    # interaction first and then run the automatic merge:

    # Start by find all overlapping fragments
    overlapping = []
    for n, m in tqdm(zip(x, mesh), desc='Pre-processing neuron(s)',
                     leave=False, disable=not use_pbars, total=len(x)):
        ol = find_fragments(n,
                            min_node_overlap=min_node_overlap,
                            min_nodes=min_overlap_size,
                            mesh=m,
                            remote_instance=target_instance)

        if ol:
            # Add number of samplers to each neuron
            n_samplers = pymaid.get_sampler_counts(ol,
                                                   remote_instance=target_instance)

            for nn in ol:
                nn.sampler_count = n_samplers[str(nn.id)]

        overlapping.append(ol)

    # Now have the user confirm merges before we actually make them
    viewer = navis.Viewer(title='Confirm merges')
    viewer.clear()
    overlap_cnf = []
    base_neurons = []
    try:
        for n, ol in zip(x, overlapping):
            # This asks user a bunch of questions prior to merge and upload
            ol, bn = confirm_overlap(n, ol, viewer=viewer)
            overlap_cnf.append(ol)
            base_neurons.append(bn)
    except BaseException:
        raise
    finally:
        viewer.close()

    for i, (n, ol, bn, m) in enumerate(zip(x, overlap_cnf, base_neurons, mesh)):
        print(f'Processing neuron "{n.name}" ({n.id}) [{i}/{len(x)}]', flush=True)
        # If no overlapping neurons proceed with just uploading.
        if not ol:
            print('No overlapping fragments found. Uploading without merging...',
                  end='', flush=True)
            resp = pymaid.upload_neuron(n,
                                        import_tags=import_tags,
                                        import_annotations=True,
                                        import_connectors=True,
                                        remote_instance=target_instance)
            if 'error' in resp:
                return resp

            # Add annotations
            _ = _merge_annotations(n, resp['skeleton_id'], tag, target_instance)

            msg = '\nNeuron "{}" successfully uploaded to target instance as "{}" #{}'
            print(msg.format(n.name, n.name, resp['skeleton_id']),
                  flush=True)
            continue

        # Check if there is a duplicate skeleton ID between the to-be-merged
        # neuron and the to-merge-into neurons
        original_skid = None
        if n.id in ol.id:
            print('Fixing duplicate skeleton IDs.',
                  flush=True)
            # Keep track of old skid
            original_skid = n.id
            # Skeleton ID must stay convertable to integer
            n.id = str(random.randint(1, 1000000))
            n._clear_temp_attr()

        # Check if there are any duplicate node IDs between neuron ``x`` and the
        # overlapping fragments and create new IDs for ``x`` if necessary
        duplicated = n.nodes[n.nodes.node_id.isin(ol.nodes.node_id.values)]
        if not duplicated.empty:
            print('Duplicate node IDs found. Regenerating node tables... ',
                  end='', flush=True)
            max_ix = max(ol.nodes.node_id.max(), n.nodes.node_id.max()) + 1
            new_ids = range(max_ix, max_ix + duplicated.shape[0])
            id_map = {old: new for old, new in zip(duplicated.node_id, new_ids)}
            n.nodes['node_id'] = n.nodes.node_id.map(lambda n: id_map.get(n, n))
            n.nodes['parent_id'] = n.nodes.parent_id.map(lambda n: id_map.get(n, n))
            if n.has_connectors:
                n.connectors['node_id'] = n.connectors.node_id.map(lambda n: id_map.get(n, n))
            n._clear_temp_attr()
            print('Done.', flush=True)

        # Combining the fragments into a single neuron is actually non-trivial:
        # 1. Collapse nodes of our input neuron `x` into within-distance nodes
        #    in the overlapping fragments (never the other way around!)
        # 2. At the same time keep connectivity (i.e. edges) of the input-neuron
        # 3. Keep track of the nodes' provenance (i.e. the contractions)
        #
        # In addition there are a lot of edge-cases to consider. For example:
        # - multiple nodes collapsing onto the same node
        # - nodes of overlapping fragments that are close enough to be collapsed
        #   (e.g. orphan synapse nodes)

        # Keep track of original skeleton IDs
        for a in ol + n:
            # Original skeleton of each node
            a.nodes['origin_skeletons'] = a.id
            if a.has_connectors:
                # Original skeleton of each connector
                a.connectors['origin_skeletons'] = a.id

        print('Generating union of all fragments... ', end='', flush=True)
        union, new_edges, collapsed_into = collapse_nodes(n, ol,
                                                          limit=merge_limit,
                                                          base_neuron=bn,
                                                          mesh=m)
        print('Done.', flush=True)

        print('Extracting new nodes to upload... ', end='', flush=True)
        # Now we have to break the neuron into "new" fragments that we can upload
        # First get the new and old nodes
        new_nodes = union.nodes[union.nodes.origin_skeletons == n.id].node_id.values
        old_nodes = union.nodes[union.nodes.origin_skeletons != n.id].node_id.values

        # Now remove the already existing nodes from the union
        only_new = navis.subset_neuron(union, new_nodes)

        # And then break into continuous fragments for upload
        frags = navis.break_fragments(only_new)
        print('Done.', flush=True)

        # Also get the new edges we need to generate
        to_stitch = new_edges[~new_edges.parent_id.isnull()]

        # We need this later -> no need to compute this for every uploaded fragment
        cond1b = to_stitch.node_id.isin(old_nodes)
        cond2b = to_stitch.parent_id.isin(old_nodes)

        # Now upload each fragment and keep track of new node IDs
        tn_map = {}
        for f in tqdm(frags, desc='Merging new arbors', leave=False, disable=not use_pbars):
            # In cases of complete merging into existing neurons, the fragment
            # will have no nodes
            if f.n_nodes < 1:
                continue

            # Check if fragment is a "linker" and as such can not be skipped
            lcond1 = np.isin(f.nodes.node_id.values,
                             new_edges.node_id.values)
            lcond2 = np.isin(f.nodes.node_id.values,
                             new_edges.parent_id.values)


            # If not linker, check skip conditions
            if sum(lcond1) + sum(lcond2) <= 1:
                if f.cable_length < min_upload_size:
                    continue
                if f.n_nodes < min_upload_nodes:
                    continue

            # Collect origin info for this neuron if it's a CatmaidNeuron
            if isinstance(n, pymaid.CatmaidNeuron):
                source_info = {'source_type': 'segmentation'}

                if not sid_from_nodes or 'origin_skeletons' not in f.nodes.columns:
                    # If we had to change the skeleton ID due to duplication, make
                    # sure to pass the original skid as source ID
                    if original_skid:
                        source_info['source_id'] = int(original_skid)
                    else:
                        source_info['source_id'] = int(n.id)
                else:
                    if f.nodes.origin_skeletons.unique().shape[0] == 1:
                        skid = f.nodes.origin_skeletons.unique()[0]
                    else:
                        print('Warning: uploading chimera fragment with multiple '
                              'skeleton IDs! Using largest contributor ID.')
                        # Use the skeleton ID that has the most nodes
                        by_skid = f.nodes.groupby('origin_skeletons').x.count()
                        skid = by_skid.sort_values(ascending=False).index.values[0]

                    source_info['source_id'] = int(skid)

                if not isinstance(getattr(n, '_remote_instance', None), type(None)):
                    source_info['source_project_id'] = n._remote_instance.project_id
                    source_info['source_url'] = n._remote_instance.server
            else:
                # Unknown source
                source_info = {}

            resp = pymaid.upload_neuron(f,
                                        import_tags=import_tags,
                                        import_annotations=False,
                                        import_connectors=True,
                                        remote_instance=target_instance,
                                        **source_info)

            # Stop if there was any error while uploading
            if 'error' in resp:
                return resp

            # Collect old -> new node IDs
            tn_map.update(resp['node_id_map'])

            # Now check if we can create any of the new edges by joining nodes
            # Both treenode and parent ID have to be either existing nodes or
            # newly uploaded
            cond1a = to_stitch.node_id.isin(tn_map)
            cond2a = to_stitch.parent_id.isin(tn_map)

            to_gen = to_stitch.loc[(cond1a | cond1b) & (cond2a | cond2b)]

            # Join nodes
            for node in to_gen.itertuples():
                # Make sure our base_neuron always come out as winner on top
                if node.node_id in bn.nodes.node_id.values:
                    winner, looser = node.node_id, node.parent_id
                else:
                    winner, looser = node.parent_id, node.node_id

                # We need to map winner and looser to the new node IDs
                winner = tn_map.get(winner, winner)
                looser = tn_map.get(looser, looser)

                # And now do the join
                resp = pymaid.join_nodes(winner,
                                         looser,
                                         no_prompt=True,
                                         tag_nodes=label_joins,
                                         remote_instance=target_instance)

                # See if there was any error while uploading
                if 'error' in resp:
                    print('Skipping joining nodes '
                          '{} and {}: {} - '.format(node.node_id,
                                                    node.parent_id,
                                                    resp['error']))
                    # Skip changing confidences
                    continue

                # Pop this edge from new_edges and from condition
                new_edges.drop(node.Index, inplace=True)
                cond1b.drop(node.Index, inplace=True)
                cond2b.drop(node.Index, inplace=True)

                # Change node confidences at new join
                if label_joins:
                    new_conf = {looser: 1}
                    resp = pymaid.update_node_confidence(new_conf,
                                                         remote_instance=target_instance)

        # Add annotations
        _ = _merge_annotations(n, bn, tag, target_instance)

        # Update node radii
        if update_radii and 'radius' in n.nodes.columns and np.all(n.nodes.radius):
            print('Updating radii of existing nodes... ', end='', flush=True)
            resp = update_node_radii(source=n, target=ol,
                                     remote_instance=target_instance,
                                     limit=merge_limit,
                                     skip_existing=True)
            print('Done.', flush=True)

        print('Neuron "{}" successfully merged into target instance as "{}" #{}'.format(n.name, bn.name, bn.id),
              flush=True)

    return


def _merge_annotations(n, bn, tag, target_instance):
    """Make sure proper annotations are added."""
    to_add = []
    # Add "{URL} upload {tag} annotation"
    if not isinstance(getattr(n, '_remote_instance', None), type(None)):
        u = n._remote_instance.server.split('/')[-1] + ' upload'
        if isinstance(tag, str):
            u += " {}".format(tag)
        to_add.append(u)
    # Existing annotation (the individual fragments would not have inherited them)
    if n.__dict__.get('_annotations', []):
        to_add += n._annotations
    # If anything to add
    if to_add:
        _ = pymaid.add_annotations(bn,
                                   to_add,
                                   remote_instance=target_instance)


def update_node_radii(source, target, remote_instance, limit=2, skip_existing=True):
    """Update node radii in target neuron from their nearest neighbor in source neuron.

    Parameters
    ----------
    source :            CatmaidNeuron
                        Neuron which node radii to use to update target neuron.
    target :            CatmaidNeuron
                        Neuron which node radii to update.
    remote_instance :   CatmaidInstance
                        Catmaid instance in which ``target`` lives.
    limit :             int, optional
                        Max distance [um] between source and target neurons for
                        nearest neighbor search.
    skip_existing :     bool, optional
                        If True, will skip nodes in ``source`` that already have
                        a radius >0.

    Returns
    -------
    dict
                        Server response.

    """
    if not isinstance(source, (navis.TreeNeuron, navis.NeuronList)):
        raise TypeError('Expected navis.TreeNeuron, pymaid.CatmaidNeuron '
                        'or NeuronList, got "{}"'.format(type(source)))

    if not isinstance(target, (navis.TreeNeuron, navis.NeuronList)):
        raise TypeError('Expected navis.TreeNeuron, pymaid.CatmaidNeuron '
                        'or NeuronList, got "{}"'.format(type(target)))

    # Turn limit from microns to nanometres
    limit *= 1000

    # First find the closest neighbor within distance limit for each node in target
    # Find nodes in A to be merged into B
    tree = navis.neuron2KDTree(source, tree_type='c', data='nodes')

    nodes = target.nodes
    if skip_existing:
        # Extract nodes without a radius
        nodes = nodes[nodes.radius <= 0]

    # For each node in A get the nearest neighbor in B
    coords = nodes[['x', 'y', 'z']].values
    nn_dist, nn_ix = tree.query(coords, k=1, distance_upper_bound=limit)

    # Find nodes that are close enough to collapse
    tn_ids = nodes.loc[nn_dist <= limit].node_id.values
    new_radii = source.nodes.iloc[nn_ix[nn_dist <= limit]].radius.values

    return pymaid.update_radii(dict(zip(tn_ids, new_radii)),
                               remote_instance=remote_instance)
