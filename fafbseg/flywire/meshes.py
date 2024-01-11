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
from concurrent.futures import ThreadPoolExecutor, as_completed

from .l2 import get_l2_graph
from .synapses import get_synapses
from .utils import get_cloudvolume, inject_dataset
from .annotations import parse_neuroncriteria

__all__ = ["get_mesh_neuron"]


@parse_neuroncriteria()
@inject_dataset()
def get_mesh_neuron(
    id,
    with_synapses=False,
    omit_failures=None,
    threads=5,
    lod=None,
    progress=True,
    *,
    dataset=None,
):
    """Fetch FlyWire neuron as navis.MeshNeuron.

    Parameters
    ----------
    id  :               int | list of int | NeuronCriteria
                        Root ID(s) to fetch meshes for.
    with_synapses :     bool
                        If True, will also load a connector table with
                        synapse predicted by Buhmann et al. (2020). This uses
                        the default parameters for ``flywire.get_synapses``.
                        Use that function directly (with ``attach=True``) to
                        have more control over which synapses get pulled.
    omit_failures :     bool, optional
                        Determine behaviour when mesh download fails
                        (e.g. if there is no mesh):
                         - ``None`` (default) will raise an exception
                         - ``True`` will skip the offending neuron (might result
                           in an empty ``NeuronList``)
                         - ``False`` will return an empty ``MeshNeuron``
    threads :           bool | int, optional
                        Whether to use threads to fetch meshes in parallel.
    lod :               int [0-3], optional
                        Level-of-detail; higher = lower resolution. Only
                        relevant if dataset is actually has multi-resolution
                        meshes. Defaults to 2. Note that not all meshes are
                        available at the lowest resolution (lod 3). If that
                        happens, we will automatically try to fetch the next
                        lowest one.
    dataset :           str | CloudVolume
                        Against which FlyWire dataset to query::
                          - "production" (currently fly_v31)
                          - "public"
                          - "sandbox" (currently fly_v26)
                          - "flat_630" or "flat_571" will use the flat
                            segmentations matching the respective materialization
                            versions. By default these use `lod=2`, you can
                            change that behaviour by passing `lod` as keyword
                            argument.
                            If ``None`` will fall back to the default dataset
                            (see :func:`~fafbseg.flywire.set_default_dataset`).

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
        raise ValueError(
            "`omit_failures` must be either None, True or False. "
            f'Got "{omit_failures}".'
        )

    vol = get_cloudvolume(dataset)

    if navis.utils.is_iterable(id):
        id = np.asarray(id, dtype=np.int64)
        if 0 in id:
            raise ValueError("Root ID 0 among the queried IDs")

        if not threads or threads == 1:
            return navis.NeuronList(
                [
                    get_mesh_neuron(
                        n,
                        dataset=dataset,
                        omit_failures=omit_failures,
                        lod=lod,
                        with_synapses=with_synapses,
                    )
                    for n in navis.config.tqdm(
                        id, desc="Fetching", disable=not progress, leave=False
                    )
                ]
            )
        else:
            if not isinstance(threads, int):
                raise TypeError(
                    f'`threads` must be int or `None`, got "{type(threads)}".'
                )
            with ThreadPoolExecutor(max_workers=threads) as executor:
                futures = {
                    executor.submit(
                        get_mesh_neuron,
                        n,
                        dataset=dataset,
                        omit_failures=omit_failures,
                        threads=None,  # no need for threads in inner function
                        with_synapses=with_synapses,
                    ): n
                    for n in id
                }

                results = []
                for f in navis.config.tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc="Fetching",
                    disable=not progress,
                    leave=False,
                ):
                    results.append(f.result())
            return navis.NeuronList(results)

    # Make sure the ID is integer
    id = np.int64(id)

    # Fetch mesh
    try:
        old_parallel = vol.parallel  # note: this seems to be reset by cloudvolume
        vol.parallel = threads if threads else old_parallel
        if vol.path.startswith("graphene"):
            mesh = vol.mesh.get(id, deduplicate_chunk_boundaries=False)[id]
        elif vol.path.startswith("precomputed"):
            lod_ = lod if lod is not None else 2
            while lod_ >= 0:
                try:
                    mesh = vol.mesh.get(id, lod=lod_)[id]
                    break
                except cv.exceptions.MeshDecodeError:
                    lod_ -= 1
                except BaseException:
                    raise
            if lod_ < 0:
                raise ValueError(f"No mesh for id {id} found")
    except KeyboardInterrupt:
        raise
    except BaseException:
        if omit_failures is None:
            raise
        elif omit_failures:
            return navis.NeuronList([])
        # If no omission, return empty MeshNeuron
        else:
            return navis.MeshNeuron(None, id=id, units="1 nm", dataset=dataset)
    finally:
        vol.parallel = old_parallel

    # Turn into meshneuron
    n = navis.MeshNeuron(mesh, id=id, units="nm", dataset=dataset)

    if with_synapses:
        _ = get_synapses(
            n.id, attach=True, min_score=30, dataset=dataset, progress=False
        )

    return n


def _get_mesh(seg_id, vol):
    """Fetch mesh."""
    import DracoPy

    level = vol.mesh.meta.meta.decode_layer_id(seg_id)
    fragment_filenames = vol.mesh.get_fragment_filenames(
        seg_id, level=level, bbox=None, bypass=False
    )
    fragments = vol.mesh._get_mesh_fragments(fragment_filenames)
    fragments = sorted(
        fragments, key=lambda frag: frag[0]
    )  # make decoding deterministic

    for i, (filename, frag) in enumerate(fragments):
        mesh = None

        if frag is not None:
            try:
                # Easier to ask forgiveness than permission
                mesh = Mesh.from_draco(frag)
            except DracoPy.FileTypeException:
                mesh = Mesh.from_precomputed(frag)

        mesh.segid = np.int64(filename.split("/")[1].split(":")[0])

        fragments[i] = mesh

    fragments = [f for f in fragments if f is not None]
    if len(fragments) == 0:
        raise IndexError("No mesh fragments found for segment {}".format(seg_id))

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

    G = get_l2_graph(seg_id)

    edges = [e for e in G.edges if e[0] in l2_map and e[1] in l2_map]
    edges = [[l2_map[e[0]], l2_map[e[1]]] for e in edges]
    edges = [e for e in edges if e[0] != e[1]]

    return mesh, l2_map


def detect_soma(x, min_rad=800, N=3, progress=True, *, dataset=None):
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
    dataset :   "public" | "production" | "sandbox" | "flat_630", optional
                Against which FlyWire dataset to query. If ``None`` will fall
                back to the default dataset (see
                :func:`~fafbseg.flywire.set_default_dataset`).

    Returns
    -------
    np.array
                Either (3, ) if input is single neuron or (N, 3) if input
                is N neurons/ids. If no soma is found, will return
                coordinates [None, None, None].

    """
    if navis.utils.is_iterable(x):
        return np.vstack(
            [
                detect_soma(n, dataset=dataset)
                for n in navis.config.tqdm(
                    x, desc="Detecting", disable=not progress, leave=False
                )
            ]
        )

    if not isinstance(x, (tm.Trimesh, navis.MeshNeuron)):
        x = get_mesh_neuron(x, dataset=dataset).trimesh
    elif isinstance(x, navis.MeshNeuron):
        x = x.trimesh

    centers, radii, G = sk.skeletonize.wave._cast_waves(
        x, waves=3, step_size=1, progress=True
    )

    is_big = np.where(radii >= min_rad)[0]

    if not any(is_big):
        return np.array([None, None, None])

    # Find stretches of consectutive above-threshold radii
    candidates = []
    for stretch in np.split(is_big, np.where(np.diff(is_big) != 1)[0] + 1):
        if len(stretch) < N:
            continue
        candidates += [i for i in stretch]

    if not candidates:
        return np.array([None, None, None])

    # Find the largest candidate
    candidates = sorted(candidates, key=lambda x: radii[x])

    return centers[candidates[-1]]


@inject_dataset()
def mesh_neuron(x, mip=2, thin=False, bounds=None, progress=True, *, dataset=None):
    """Create high quality mesh for given neuron.

    Parameters
    ----------
    x :         int
                Single FlyWire root ID.
    mip :       int [0-3]
                Level of detail. Lower = higher resolution.
    bounds :    (3, 2) or (2, 3) array, optional
                Bounding box to return voxels in. Expected to be in 4, 4, 40
                voxel space.
    thin :      bool
                If True, will remove voxels at the interface of adjacent
                supervoxels that are not supposed to be connected according
                to the L2 graph. This is rather expensive but can help in
                situations where a neuron self-touches.
    dataset :   "public" | "production" | "sandbox" | "flat_630", optional
                Against which FlyWire dataset to query. If ``None`` will fall
                back to the default dataset (see
                :func:`~fafbseg.flywire.set_default_dataset`).

    Returns
    -------
    navis.MeshNeuron

    """
    try:
        import sparsecubes as sc
    except ImportError:
        raise ImportError("Meshing requires sparse-cubes:\n  pip3 install sparse-cubes")

    from .segmentation import get_voxels

    # Get voxels for this neuron
    vxl = get_voxels(
        x, mip=mip, thin=thin, bounds=bounds, progress=progress, dataset=dataset
    )

    vol = get_cloudvolume(dataset)
    spacing = vol.scales[mip]["resolution"]

    mesh = sc.marching_cubes(vxl, spacing=spacing)

    return navis.MeshNeuron(mesh, id=x, units="1 nm")
