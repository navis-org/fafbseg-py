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

import cloudvolume
import mcubes
import numpy as np
import tqdm
import pymaid

from pyoctree import pyoctree

from . import utils, segmentation

use_pbars = utils.use_pbars
CVtype = cloudvolume.frontends.precomputed.CloudVolumePrecomputed


def get_mesh(x, bbox, vol=None):
    """Get mesh for given segmentation ID using CloudVolume.

    Parameters
    ----------
    x :     int, list of ints
            Segmentation ID(s).
    bbox :  list-like
            Bounding box. Either ``[x1, x2, y1, y2, z1, z2]`` or
            ``[[x1, x2], [y1, y2], [z1, z2]]``. Coordinates are expected to
            be in nanometres.
    vol :   cloudvolume.CloudVolume

    Returns
    -------
    pymaid.Volume

    None
            If queried ID(s) are either only in a single voxel or if there are
            no IDs other than the queried ID(s) in the bounding box.

    """
    if not isinstance(x, (list, np.ndarray)):
        x = [x]

    if isinstance(vol, type(None)):
        vol = getattr(utils, 'vol')

    assert isinstance(vol, CVtype)

    if not isinstance(vol, CVtype):
        raise TypeError('Expected CloudVolume, got "{}"'.format(type(vol)))

    # Parse bbox
    bbox = np.array(bbox).reshape(3, 2)
    # Convert to pixel coords
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
        #print('ID(s) are in a single voxel.')
        return None
    elif n_voxels == np.prod(u.shape):
        #print('All voxels in cutout are queried segmentation IDs')
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

    # Turn into pymaid Volume and return
    return pymaid.Volume(verts, faces, name=str(x))


def test_edges(neuron, edges=None, vol=None):
    """Test if edge(s) cross membranes using ray-casting.

    Parameters
    ----------
    neuron :    pymaid.CatmaidNeuron
                Neuron to test edges for.
    edges :     list-like, optional
                Edges to test. Can be:
                 1. List of single treenode IDs
                 2. List of pairs of treenode IDs
                 3. ``None`` in which case all edges will be tested.
    vol :       cloudvolume.CloudVolume

    Returns
    -------
    array
                Array containing True/False for each tested each.

    """
    if isinstance(vol, type(None)):
        vol = getattr(utils, 'vol')

    assert isinstance(vol, CVtype)
    assert isinstance(neuron, pymaid.CatmaidNeuron)

    if isinstance(edges, type(None)):
        not_root = ~neuron.nodes.parent_id.isnull()
        edges = neuron.nodes[not_root].treenode_id.values

    edges = np.array(edges)

    nodes = neuron.nodes.set_index('treenode_id')
    if edges.ndim == 1:
        locs1 = nodes.loc[edges][['x', 'y', 'z']].values
        parents = nodes.loc[edges].parent_id.values
        locs2 = nodes.loc[parents][['x', 'y', 'z']].values
    elif edges.ndim == 2:
        locs1 = nodes.loc[edges[:, 0]][['x', 'y', 'z']].values
        locs2 = nodes.loc[edges[:, 1]][['x', 'y', 'z']].values
    else:
        raise ValueError('Unexpected format for edges: {}'.format(edges.shape))

    # Get the segmentation IDs
    segids1 = segmentation.get_seg_ids(locs1)
    segids2 = segmentation.get_seg_ids(locs2)

    # Now iterate over each edge
    verdict = []
    for l1, l2, s1, s2 in tqdm.tqdm(zip(locs1, locs2, segids1, segids2),
                                    total=len(locs1),
                                    desc='Checking edges'):

        # Get the bounding box
        bbox = np.array([l1, l2])
        bbox = np.array([bbox.min(axis=0), bbox.max(axis=0)]).T

        # Get the mesh
        mesh = get_mesh(s1, bbox=bbox, vol=vol)

        # No mesh means that edge is most likely True
        if not mesh:
            verdict.append(True)
            continue

        # Prepare raycasting
        tree = pyoctree.PyOctree(np.array(mesh.vertices,
                                          dtype=float, order='C'),
                                 np.array(mesh.faces,
                                          dtype=np.int32, order='C')
                                 )

        # Generate raypoints
        rayp = np.array([l1, l2], dtype=np.float32)

        # Get intersections and extract coordinates of intersection
        inters = np.array([i.p for i in tree.rayIntersection(rayp)])

        # If not intersections treat this edge as True
        if not inters.any():
            verdict.append(True)
            continue

        # In a few odd cases we can get the multiple intersections at the
        # exact same coordinate (something funny with the faces)
        unique_int = np.unique(np.round(inters), axis=0)

        # Rays are bidirectional and travel infinitely -> we have to filter
        # for those that occure between the points
        minx, miny, minz = np.min(rayp, axis=0)
        maxx, maxy, maxz = np.max(rayp, axis=0)

        cminx = (unique_int[:, 0] >= minx)
        cmaxx = (unique_int[:, 0] <= maxx)
        cminy = (unique_int[:, 1] >= miny)
        cmaxy = (unique_int[:, 1] <= maxy)
        cminz = (unique_int[:, 2] >= minz)
        cmaxz = (unique_int[:, 2] <= maxz)

        all_cond = cminx & cmaxx & cminy & cmaxy & cminz & cmaxz

        unilat_int = unique_int[all_cond]

        if unilat_int.any():
            verdict.append(False)
        else:
            verdict.append(True)

    return verdict
