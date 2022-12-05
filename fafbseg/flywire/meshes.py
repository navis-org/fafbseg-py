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

import numpy as np
import skeletor as sk
import trimesh as tm
import cloudvolume as cv

from cloudvolume.mesh import Mesh
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from .l2 import l2_graph, chunks_to_nm
from .synapses import fetch_synapses
from .utils import parse_volume

__all__ = ['get_mesh_neuron']


def get_mesh_neuron(id, with_synapses=False, omit_failures=None, threads=2,
                    progress=True, dataset='production'):
    """Fetch FlyWire neuron as navis.MeshNeuron.

    Parameters
    ----------
    id  :               int | list of int
                        Segment ID(s) to fetch meshes for.
    with_synapses :     bool
                        If True, will also load a connector table with
                        synapse predicted by Buhmann et al. (2020).
                        A "synapse score" (confidence) threshold of 30 is
                        applied.
    omit_failures :     bool, optional
                        Determine behaviour when mesh download fails
                        (e.g. if there is no mesh):
                         - ``None`` (default) will raise an exception
                         - ``True`` will skip the offending neuron (might result
                           in an empty ``NeuronList``)
                         - ``False`` will return an empty ``MeshNeuron``
    threads :           bool | int, optional
                        Whether to use threads to fetch meshes in parallel.
    dataset :           str | CloudVolume
                        Against which FlyWire dataset to query::
                          - "production" (currently fly_v31)
                          - "sandbox" (currently fly_v26)

    Return
    ------
    navis.MeshNeuron

    Examples
    --------
    >>> from fafbseg import flywire
    >>> m = flywire.get_mesh_neuron(720575940614131061)
    >>> m.plot3d()                                              # doctest: +SKIP

    """
    if omit_failures not in (None, True, False):
        raise ValueError('`omit_failures` must be either None, True or False. '
                         f'Got "{omit_failures}".')

    vol = parse_volume(dataset)

    if navis.utils.is_iterable(id):
        id = np.asarray(id, dtype=np.int64)
        if 0 in id:
            raise ValueError('Root ID 0 among the queried IDs')

        if not threads or threads == 1:
            return navis.NeuronList([get_mesh_neuron(n, dataset=dataset,
                                                     omit_failures=omit_failures,
                                                     with_synapses=with_synapses)
                                     for n in navis.config.tqdm(id,
                                                                desc='Fetching',
                                                                disable=not progress,
                                                                leave=False)])
        else:
            if not isinstance(threads, int):
                raise TypeError(f'`threads` must be int or `None`, got "{type(threads)}".')
            with ThreadPoolExecutor(max_workers=threads) as executor:
                futures = {executor.submit(get_mesh_neuron, n,
                                           dataset=dataset,
                                           omit_failures=omit_failures,
                                           threads=None,  # no need for threads in inner function
                                           with_synapses=with_synapses): n for n in id}

                results = []
                for f in navis.config.tqdm(as_completed(futures),
                                           total=len(futures),
                                           desc='Fetching',
                                           disable=not progress,
                                           leave=False):
                    results.append(f.result())
            return navis.NeuronList(results)

    # Make sure the ID is integer
    id = np.int64(id)

    # Fetch mesh
    try:
        old_parallel = vol.parallel  # note: this seems to be reset by cloudvolume
        vol.parallel = threads if threads else old_parallel
        mesh = vol.mesh.get(id, remove_duplicate_vertices=True)[id]
    except BaseException:
        if omit_failures == None:
            raise
        elif omit_failures:
            return navis.NeuronList([])
        # If no omission, return empty MeshNeuron
        else:
            return navis.MeshNeuron(None, id=id, units='1 nm', dataset=dataset, **kwargs)
    finally:
        vol.parallel = old_parallel

    # Turn into meshneuron
    n = navis.MeshNeuron(mesh, id=id, units='nm', dataset=dataset)

    if with_synapses:
        _ = fetch_synapses(n, attach=True, min_score=30, dataset=dataset,
                           progress=False)

    return n


def get_mesh_neuron_flat(id, lod=3, with_synapses=False, omit_failures=None,
                         threads=10, progress=True):
    """Fetch FlyWire neuron as navis.MeshNeuron.

    This uses the flat segmentation for materialization 526.


    Parameters
    ----------
    id  :               int | list of int
                        Segment ID(s) to fetch meshes for. Currently, the root
                        IDs must have existed at materialization 526.
    lod :               int [0-3]
                        Level-of-detail; higher = lower resolution. Note that not
                        all meshes are available at the lowest resolution (lod 3).
                        If that happens, we will automatically try to fetch the
                        next lowest one.
    with_synapses :     bool
                        If True, will also load a connector table with
                        synapse predicted by Buhmann et al. (2020).
                        A "synapse score" (confidence) threshold of 30 is
                        applied.
    omit_failures :     bool, optional
                        Determine behaviour when mesh download fails
                        (e.g. if there is no mesh):
                         - ``None`` (default) will raise an exception
                         - ``True`` will skip the offending neuron (might result
                           in an empty ``NeuronList``)
                         - ``False`` will return an empty ``MeshNeuron``
    threads :           bool | int, optional
                        Whether to use threads to fetch meshes in parallel.

    Return
    ------
    navis.MeshNeuron

    """
    if omit_failures not in (None, True, False):
        raise ValueError('`omit_failures` must be either None, True or False. '
                         f'Got "{omit_failures}".')

    vol = cv.CloudVolume('precomputed://gs://flywire_v141_m526',
                         use_https=True, parallel=False,
                         progress=False, cache=False)

    if navis.utils.is_iterable(id):
        id = np.asarray(id, dtype=np.int64)
        if 0 in id:
            raise ValueError('Root ID 0 among the queried IDs')

        if not threads or threads == 1:
            return navis.NeuronList([get_mesh_neuron_flat(n,
                                                          lod=lod,
                                                          omit_failures=omit_failures,
                                                          with_synapses=with_synapses)
                                     for n in navis.config.tqdm(id,
                                                                desc='Fetching',
                                                                disable=not progress,
                                                                leave=False)])
        else:
            if not isinstance(threads, int):
                raise TypeError(f'`threads` must be int or `None`, got "{type(threads)}".')
            with ThreadPoolExecutor(max_workers=threads) as executor:
                futures = {executor.submit(get_mesh_neuron_flat, n,
                                           lod=lod,
                                           omit_failures=omit_failures,
                                           threads=None,  # no need for threads in inner function
                                           with_synapses=with_synapses): n for n in id}

                results = []
                for f in navis.config.tqdm(as_completed(futures),
                                           total=len(futures),
                                           desc='Fetching',
                                           disable=not progress,
                                           leave=False):
                    results.append(f.result())
            return navis.NeuronList(results)

    # Make sure the ID is integer
    id = np.int64(id)

    # Fetch mesh
    try:
        old_parallel = vol.parallel  # note: this seems to be reset by cloudvolume
        vol.parallel = threads if threads else old_parallel
        lod_ = lod
        while lod >= 0:
            try:
                mesh = vol.mesh.get(id, lod=lod_)[id]
                break
            except cv.exceptions.MeshDecodeError:
                lod_ -= 1
            except BaseException:
                if omit_failures == None:
                    raise
                elif omit_failures:
                    return navis.NeuronList([])
                # If no omission, return empty MeshNeuron
                else:
                    return navis.MeshNeuron(None, id=id, units='1 nm', **kwargs)
        if lod_ <= 0:
            if omit_failures == None:
                raise BaseException(f'No neuron with root id {id} found.')
            elif omit_failures:
                return navis.NeuronList([])
            # If no omission, return empty MeshNeuron
            else:
                return navis.MeshNeuron(None, id=id, units='1 nm',  **kwargs)
    finally:
        vol.parallel = old_parallel

    # Turn into meshneuron
    n = navis.MeshNeuron(mesh, id=id, units='nm', lod=lod_)

    if with_synapses:
        _ = fetch_synapses(n, attach=True, min_score=30, mat=526, progress=False)

    return n


def _get_mesh(seg_id, vol):
    """Fetch mesh."""
    import DracoPy

    level = vol.mesh.meta.meta.decode_layer_id(seg_id)
    fragment_filenames = vol.mesh.get_fragment_filenames(seg_id,
                                                         level=level,
                                                         bbox=None,
                                                         bypass=False)
    fragments = vol.mesh._get_mesh_fragments(fragment_filenames)
    fragments = sorted(fragments, key=lambda frag: frag[0])  # make decoding deterministic

    for i, (filename, frag) in enumerate(fragments):
        mesh = None

        if frag is not None:
            try:
                # Easier to ask forgiveness than permission
                mesh = Mesh.from_draco(frag)
            except DracoPy.FileTypeException:
                mesh = Mesh.from_precomputed(frag)

        mesh.segid = np.int64(filename.split('/')[1].split(':')[0])

        fragments[i] = mesh

    fragments = [f for f in fragments if f is not None]
    if len(fragments) == 0:
        raise IndexError('No mesh fragments found for segment {}'.format(seg_id))

    # Concatenate the mesh
    mesh = Mesh.concatenate(*fragments)
    mesh.segid = seg_id

    # Generate vertex -> L2 ID mapping
    l2_map = {}
    ix = 0
    for m in fragments:
        # Get level of this fragment
        level = vol.mesh.meta.meta.decode_layer_id(m.segid)

        # If already L2 ID just track
        n_verts = m.vertices.shape[0]
        if level == 2:
            l2_map[m.segid] = range(ix, ix + n_verts)
        else:
            # If not L2, get the L2 leaves
            l2_ids = vol.get_leaves(m.segid, vol.meta.bounds(0), 0, stop_layer=2)
            for i in l2_ids:
                l2_map[i] = range(ix, ix + n_verts)

        ix += n_verts

    G = l2_graph(seg_id)

    edges = [e for e in G.edges if e[0] in l2_map and e[1] in l2_map]
    edges = [[l2_map[e[0]], l2_map[e[1]]] for e in edges]
    edges = [e for e in edges if e[0] != e[1]]

    return mesh, l2_map


def detect_soma(x, min_rad=800, N=3, progress=True, dataset='production'):
    """Try detecting the soma based on radius of the mesh.

    Parameters
    ----------
    x :         int | trimesh.Trimesh | navis.MeshNeuron
                Either ID(s) or mesh(es). Meshes must not be downsampled.
    min_rad :   float
                Minimum radius for a node to be considered a soma candidate.
    N :         int
                Number of consecutive vertices with radius > `min_rad` in
                order to consider them soma candidates.

    Returns
    -------
    np.array
                Either (3, ) if input is single neuron or (N, 3) if input
                is N neurons/ids. If no soma is found, will return
                coordinates [None, None, None].

    """
    if navis.utils.is_iterable(x):
        return np.vstack([detect_soma(n, dataset=dataset)
                          for n in navis.config.tqdm(x,
                                                     desc='Detecting',
                                                     disable=not progress,
                                                     leave=False)])

    if not isinstance(x, (tm.Trimesh, navis.MeshNeuron)):
        x = get_mesh_neuron(x, dataset=dataset).trimesh
    elif isinstance(x, navis.MeshNeuron):
        x = x.trimesh

    centers, radii, G = sk.skeletonize.wave._cast_waves(x, waves=3, step_size=1,
                                                        progress=True)

    is_big = np.where(radii >= min_rad)[0]

    if not any(is_big):
        return np.array([None, None, None])

    # Find stretches of consectutive above-threshold radii
    candidates = []
    for stretch in np.split(is_big, np.where(np.diff(is_big) != 1)[0]+1):
        if len(stretch) < N:
            continue
        candidates += [i for i in stretch]

    if not candidates:
        return np.array([None, None, None])

    # Find the largest candidate
    candidates = sorted(candidates, key=lambda x: radii[x])

    return centers[candidates[-1]]


def mesh_neuron(x, mip=2, thin=False, bounds=None, progress=True, dataset='production'):
    """Create high quality mesh for given neuron."""
    try:
        import sparsecubes as sc
    except ImportError:
        raise ImportError('Meshing requires sparse-cubes:\n  pip3 install sparse-cubes')

    from .segmentation import get_voxels

    # Get voxels for this neuron
    vxl = get_voxels(x, mip=mip, thin=thin, bounds=bounds, progress=progress, dataset=dataset)

    vol = parse_volume(dataset)
    spacing = vol.scales[mip]['resolution']

    mesh = sc.marching_cubes(vxl, spacing=spacing)

    return navis.MeshNeuron(mesh, id=x, units='1 nm')
