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

import cloudvolume
import ncollpyde
import tqdm
import pymaid

import numpy as np
import pandas as pd

from concurrent import futures

from .. import utils
from .segmentation import locs_to_segments

try:
    import mcubes
except ImportError:
    mcubes = None
except BaseException:
    raise

use_pbars = utils.use_pbars
CVtype = cloudvolume.frontends.precomputed.CloudVolumePrecomputed

__all__ = ['get_mesh', 'autoreview_edges', 'test_edges']


def get_mesh(x, bbox, vol=None):
    """Get mesh for given segmentation ID using CloudVolume.

    This produces meshes from scratch using marching cubes. As this requires
    loading the segmentation data it is not suited for generating meshes for
    whole neurons.

    Parameters
    ----------
    x :     int, list of ints
            Segmentation ID(s). Will produce a single mesh from all
            segmentation IDs.
    bbox :  list-like
            Bounding box. Either ``[x1, x2, y1, y2, z1, z2]`` or
            ``[[x1, x2], [y1, y2], [z1, z2]]``. Coordinates are expected to
            be in nanometres.
    vol :   cloudvolume.CloudVolume
            CloudVolume pointing to the segmentation data.

    Returns
    -------
    pymaid.Volume

    None
            If queried ID(s) are either only in a single voxel or if there are
            no IDs other than the queried ID(s) in the bounding box.

    """
    if not mcubes:
        raise ImportError('Unable to import mcubes (PyMCubes) library.')

    if not isinstance(x, (list, np.ndarray)):
        x = [x]

    if isinstance(vol, type(None)):
        vol = getattr(utils, 'vol')

    assert isinstance(vol, CVtype)

    if not isinstance(vol, CVtype):
        raise TypeError('Expected CloudVolume, got "{}"'.format(type(vol)))

    # Parse bbox
    bbox = np.array(bbox).reshape(3, 2)
    # Convert to voxel coords
    bbox = bbox / np.array(vol.scale['resolution']).reshape(3, 1)
    bbox = np.array([np.floor(bbox[:, 0]), np.ceil(bbox[:, 1])]).astype(int).T

    # Add some padding - otherwise we might get [:1] slices
    bbox += [0, 1]

    x1, x2, y1, y2, z1, z2 = bbox.flatten()

    # Get this cutout
    u = vol[x1:x2, y1:y2, z1:z2][:, :, :, 0]

    # Subset to the desired ID(s)
    u = np.isin(u, x)

    n_voxels = u.sum()

    if n_voxels == 0:
        raise ValueError("Id(s) {} not found in given bounding box.".format(x))
    elif n_voxels == 1:
        #  print('ID(s) are in a single voxel.')
        return None
    elif n_voxels == np.prod(u.shape):
        #  print('All voxels in cutout are queried segmentation IDs')
        return None

    # We have to add some padding otherwise the marching cube will fail
    u = np.pad(u, 1, mode='constant', constant_values=False)

    # Update bounding box to match padding
    bbox[:, 0] -= 1

    # Generate faces and vertices
    verts, faces = mcubes.marching_cubes(u, 0)

    # Bring into the correct coordinate space
    verts *= vol.scale['resolution']
    verts += bbox[:, 0] * vol.scale['resolution']

    # Turn into and return pymaid Volume
    return pymaid.Volume(verts, faces, name=str(x))


def autoreview_edges(x, conf_threshold=1, vol=None, remote_instance=None):
    """Automatically review (low-confidence) edges between nodes.

    The way this works:
      1. Fetch the live version of the neuron(s) from the CATMAID instance
      2. Use raycasting to test (low-confidence) edges
      3. Edge confidence is set to ``5`` if test is passed and to ``1`` if not

    You *can* use this function to test all edges in a neuron by increasing
    ``conf_threshold`` to 5. Please note that this could produce a lot of false
    positives (i.e. edges will be flagged as incorrect even though they
    aren't). Part of the problem is that mitochondria are segmented as
    separate entities and hence introduce membranes inside a neuron.

    Parameters
    ----------
    x :                 skeleton ID(s) | pymaid.CatmaidNeuron/List
                        Neuron(s) to review.
    conf_threshold :    int, optional
                        Confidence threshold for edges to be tested. By
                        default only reviews edges with confidence <= 1.
    vol :               cloudvolume.CloudVolume, optional
                        CloudVolume pointing to segmentation data.
    remote_instance :   pymaid.CatmaidInstance, optional
                        CATMAID instance. If ``None``, will use globally
                        define instance.

    Returns
    -------
    server response
                        CATMAID server response from updating node
                        confidences.

    See Also
    --------
    :func:`fafbseg.test_edges`
                        If you only need to test without changing confidences.

    Examples
    --------
    >>> # Set up CloudVolume from the publicly hosted FAFB segmentation data
    >>> # (if you have a local copy, use that instead)
    >>> from cloudvolume import CloudVolume
    >>> vol = CloudVolume('https://storage.googleapis.com/fafb-ffn1-20190805/segmentation',
    ...                   cache=True,
    ...                   progress=False)
    >>> # Autoreview edges
    >>> _ = fafbseg.autoreview_edges(14401884, vol=vol, remote_instance=manual)

    """
    # Fetch neuron(s)
    n = pymaid.get_neurons(x, remote_instance=remote_instance)

    # Extract low confidence edges
    not_root = ~n.nodes.parent_id.isnull()
    is_low_conf = n.nodes.confidence <= conf_threshold
    to_test = n.nodes[is_low_conf & not_root]

    if to_test.empty:
        print('No low-confidence edges to test in neuron(s) '
              '{} found'.format(n.skeleton_id))
        return

    # Test edges
    verdict = test_edges(n,
                         edges=to_test[['treenode_id', 'parent_id']].values,
                         vol=vol)

    # Update node confidences
    new_confidences = {n: 5 for n in to_test[verdict].treenode_id.values}
    new_confidences.update({n: 1 for n in to_test[~verdict].treenode_id.values})
    resp = pymaid.update_node_confidence(new_confidences,
                                         remote_instance=remote_instance)

    msg = '{} of {} tested low-confidence edges were found to be correct.'
    msg = msg.format(sum(verdict), to_test.shape[0])
    print(msg)

    return resp


def test_edges(x, edges=None, vol=None, max_workers=4):
    """Test if edge(s) cross membranes using ray-casting.

    Parameters
    ----------
    x :             pymaid.CatmaidNeuron | pandas.DataFrame
                    Neuron or treenode table to test edges for.
    edges :         list-like, optional
                    Use to subset to given edges. Can be:
                     1. List of single treenode IDs
                     2. List of pairs of treenode IDs
                     3. ``None`` in which case all edges will be tested. This
                        excludes the root node as it doesn't have an edge!
    vol :           cloudvolume.CloudVolume
                    CloudVolume pointing to segmentation data.
    max_workers :   int, optional
                    Maximum number of parallel worker processes to test edges.


    Returns
    -------
    numpy.ndarray
                (N, ) array containing True/False for each tested edge.

    See Also
    --------
    :func:`fafbseg.autoreview_edges`
                Use if you want to automatically review low confidence edges.

    """
    if isinstance(vol, type(None)):
        vol = getattr(utils, 'vol')

    assert isinstance(vol, CVtype)

    if isinstance(x, pymaid.CatmaidNeuron):
        nodes = x.nodes
    elif isinstance(x, pd.DataFrame):
        nodes = x
    else:
        raise TypeError('Expected CatmaidNeuron or DataFrame,'
                        ' got "{}"'.format(type(x)))

    if isinstance(edges, type(None)):
        not_root = ~nodes.parent_id.isnull()
        edges = nodes[not_root].treenode_id.values

    edges = np.array(edges)

    nodes = nodes.set_index('treenode_id', inplace=False)
    if edges.ndim == 1:
        locs1 = nodes.loc[edges][['x', 'y', 'z']].values
        parents = nodes.loc[edges].parent_id.values
        locs2 = nodes.loc[parents][['x', 'y', 'z']].values
    elif edges.ndim == 2:
        locs1 = nodes.loc[edges[:, 0]][['x', 'y', 'z']].values
        locs2 = nodes.loc[edges[:, 1]][['x', 'y', 'z']].values
    else:
        raise ValueError('Unexpected format for edges: {}'.format(edges.shape))

    # Get the segmentation IDs at the first location
    segids1 = locs_to_segments(locs1)
    with tqdm.tqdm(total=len(segids1), desc='Testing edges') as pbar:
        with futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
            point_futures = [ex.submit(_test_single_edge, *k, vol=vol) for k in zip(locs1,
                                                                                    locs2,
                                                                                    segids1)]
            for f in futures.as_completed(point_futures):
                pbar.update(1)

    return np.array([f.result() for f in point_futures])


def _test_single_edge(l1, l2, seg_id, vol):
    """Test single edge.

    Parameters
    ----------
    l1, l2 :    int | float
                Locations of the nodes connected by the edge
    seg_id :    int
                Segment ID of one of the nodes.
    vol :       cloudvolume.CloudVolume

    Returns
    -------
    bool

    """
    # Get the bounding box
    bbox = np.array([l1, l2])
    bbox = np.array([bbox.min(axis=0), bbox.max(axis=0)]).T

    # Get the mesh
    mesh = get_mesh(seg_id, bbox=bbox, vol=vol)

    # No mesh means that edge is most likely True
    if not mesh:
        return True

    # Prepare raycasting
    coll = ncollpyde.Volume(np.array(mesh.vertices, dtype=float, order='C'),
                            np.array(mesh.faces, dtype=np.int32, order='C'))

    # Get intersections
    l1 = l1.reshape(1, 3)
    l2 = l2.reshape(1, 3)
    inter_ix, inter_xyz, is_inside = coll.intersections(l1, l2)

    # If not intersections treat this edge as True
    if not inter_xyz.any():
        return True

    return True
