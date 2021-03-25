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

from .synapses import fetch_synapses
from .utils import parse_volume

__all__ = ['get_mesh_neuron']


def get_mesh_neuron(id, with_synapses=False, dataset='production'):
    """Fetch flywire neuron as navis.MeshNeuron.

    Parameters
    ----------
    id  :                int | list of int
                         Segment ID(s) to fetch meshes for.
    with_synapses :      bool
                         If True, will also load a connector table with
                         synapse predicted by Buhmann et al. (2020).
                         A "synapse score" (confidence) threshold of 30 is
                         applied.
    dataset :            str | CloudVolume
                         Against which flywire dataset to query::
                           - "production" (currently fly_v31)
                           - "sandbox" (currently fly_v26)

    Return
    ------
    navis.MeshNeuron

    Examples
    --------
    >>> from fafbseg import flywire
    >>> m = flywire.get_mesh_neuron(720575940614131061)
    >>> m.plot3d()  # doctest: +SKIP

    """
    vol = parse_volume(dataset)

    if navis.utils.is_iterable(id):
        return navis.NeuronList([get_mesh_neuron(n, dataset=dataset, with_synapses=with_synapses)
                                 for n in navis.config.tqdm(id,
                                                            desc='Fetching',
                                                            leave=False)])

    # Make sure the ID is integer
    id = int(id)

    # Fetch mesh
    mesh = vol.mesh.get(id, remove_duplicate_vertices=True)[id]

    # Turn into meshneuron
    n = navis.MeshNeuron(mesh, id=id, units='nm', dataset=dataset)

    if with_synapses:
        _ = fetch_synapses(n, attach=True, min_score=30, dataset=dataset,
                           progress=False)

    return n
