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

from tqdm.auto import tqdm

from .utils import parse_volume

__all__ = ['get_mesh_neuron']


def get_mesh_neuron(id, dataset='production'):
    """Fetch flywire neuron as navis.MeshNeuron.

    Parameters
    ----------
    id  :                int | list of int
                         Segment ID(s) to fetch meshes for.
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
    >>> m.plot3d()  # doctest

    """
    vol = parse_volume(dataset)

    if navis.utils.is_iterable(id):
        return navis.NeuronList([get_mesh_neuron(n, dataset=dataset)
                                 for n in tqdm(id, desc='Fetching', leave=False)])

    # Make sure the ID is integer
    id = int(id)

    # Fetch mesh
    mesh = vol.mesh.get(id, remove_duplicate_vertices=True)[id]

    # Turn into meshneuron
    return navis.MeshNeuron(mesh, id=id, units='nm', dataset=dataset)
