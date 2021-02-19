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

"""Functions to extract skeletons from L2 graphs.

Heavily borrows from code from Casey Schneider-Mizell's "pcg_skel"
(https://github.com/AllenInstitute/pcg_skel).

"""

import navis

import fastremap

import numpy as np
import skeletor as sk
import trimesh as tm

from annotationframeworkclient import FrameworkClient
from concurrent.futures import ThreadPoolExecutor

from .utils import parse_volume

__all__ = ['l2_skeleton']


def l2_skeleton(root_id, refine=False, drop_missing=True, threads=10,
                progress=True, dataset='production'):
    """Generate skeleton from L2 graph.

    Parameters
    ----------
    root_id  :          int
                        Root ID of the flywire neuron you want to skeletonize.
    refine :            bool
                        If True, will refine skeleton nodes by moving them in
                        the center of their corresponding chunk meshes.

    Only relevant if ``refine=True``:

    drop_missing :      bool
                        If True, will drop nodes that don't have a corresponding
                        chunk mesh. These are typically chunks that are very
                        small and dropping them might actually be benefitial.
    threads :           int
                        How many parallel threads to use for fetching the
                        chunk meshes. Reduce the number if you run into
                        ``HTTPErrors``.
    progress :          bool
                        Whether to show a progress bar.


    Return
    ------
    skeleton :          navis.TreeNeuron
                        The extraced skeleton.

    Examples
    --------
    >>> n = flywire.l2_skeleton(720575940614131061)

    """

    # Get the cloudvolume
    vol = parse_volume(dataset)

    # Hard-coded datastack names
    ds = {"production": "flywire_fafb_production",
          "sandbox": "flywire_fafb_sandbox"}
    # Note that the default server url is https://global.daf-apis.com/info/
    client = FrameworkClient(ds.get(dataset, dataset))

    # Load the L2 graph for given root ID
    # This is a (N,2) array
    lvl2_eg = np.array(client.chunkedgraph.level2_chunk_graph(root_id))

    # Drop duplicate edges
    lvl2_eg = np.unique(np.sort(lvl2_eg, axis=1), axis=0)

    # Unique L2 IDs
    lvl2_ids = np.unique(lvl2_eg)

    # ID to index
    l2dict = {l2id: ii for ii, l2id in enumerate(lvl2_ids)}

    # Remap edge graph to indices
    eg_arr_rm = fastremap.remap(lvl2_eg, l2dict)

    coords = [np.array(vol.mesh.meta.meta.decode_chunk_position(l)) for l in lvl2_ids]
    coords = np.vstack(coords)

    G = sk.skeletonizers.edges_to_graph(eg_arr_rm)
    swc = sk.skeletonizers.make_swc(G, coords=coords)

    # Convert to Eucledian space
    # Dimension of a single chunk
    ch_dims = chunks_to_nm([1, 1, 1], vol) - chunks_to_nm([0, 0, 0], vol)
    ch_dims = np.squeeze(ch_dims)

    xyz = swc[['x', 'y', 'z']].values
    swc[['x', 'y', 'z']] = chunks_to_nm(xyz, vol) + ch_dims / 2

    if refine:
        with ThreadPoolExecutor(max_workers=threads) as pool:
            futures = [pool.submit(vol.mesh.get, i,
                                   allow_missing=True,
                                   deduplicate_chunk_boundaries=False) for i in lvl2_ids]

            res = [f.result() for f in navis.config.tqdm(futures,
                                                         disable=not progress,
                                                         desc='Loading meshes')]

        # Unpack results
        meshes = {k: v for d in res for k, v in d.items()}

        # For each mesh find the center of mass and move the corresponding point
        new_co = {}
        for k, m in meshes.items():
            m = tm.Trimesh(m.vertices, m.faces)
            # Do NOT use center_mass here -> garbage if not non-watertight
            new_co[l2dict[k]] = m.centroid

        has_new = swc.node_id.isin(new_co)
        swc.loc[has_new, 'x'] = swc.loc[has_new, 'node_id'].map(lambda x: new_co[x][0])
        swc.loc[has_new, 'y'] = swc.loc[has_new, 'node_id'].map(lambda x: new_co[x][1])
        swc.loc[has_new, 'z'] = swc.loc[has_new, 'node_id'].map(lambda x: new_co[x][2])

        tn = navis.TreeNeuron(swc, id=root_id, units='1 nm')

        if drop_missing:
            tn = navis.remove_nodes(tn, swc.loc[~has_new, 'node_id'].values)
    else:
        tn = navis.TreeNeuron(swc, id=root_id, units='1 nm')

    return tn


def chunks_to_nm(xyz_ch, vol, voxel_resolution=[4, 4, 40]):
    """Map a chunk location to Euclidean space.

    Parameters
    ----------
    xyz_ch :            array-like
                        Nx3 array of chunk indices.
    cv :                cloudvolume.CloudVolume
                        CloudVolume object associated with the chunked space.
    voxel_resolution :  list, optional
                        Voxel resolution, by default [4, 4, 40].

    Returns
    -------
    np.array
                        Nx3 array of spatial points.

    """
    mip_scaling = vol.mip_resolution(0) // np.array(voxel_resolution, dtype=int)

    x_vox = np.atleast_2d(xyz_ch) * vol.mesh.meta.meta.graph_chunk_size
    return (
        (x_vox + np.array(vol.mesh.meta.meta.voxel_offset(0)))
        * voxel_resolution
        * mip_scaling
    )
