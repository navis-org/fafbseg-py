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
import trimesh as tm

from .. import utils
use_pbars = utils.use_pbars

__all__ = ['fafb14_to_flywire', 'flywire_to_fafb14']


def fafb14_to_flywire(x, coordinates='nm', mip=4, inplace=False, on_fail='warn'):
    """Transform neurons/coordinates from FAFB v14 to flywire.

    This uses a service hosted by Eric Perlman.

    Parameters
    ----------
    x :             CatmaidNeuron/List | np.ndarray (N, 3)
                    Data to transform.
    mip :           int
                    Resolution of mapping. Lower = more precise but much slower.
                    Currently only mip 4 available!
    coordinates :   "nm" | "pixel"
                    Units of the provided data in ``x``.
    inplace :       bool
                    If ``True`` will modify Neuron object(s) in place. If ``False``
                    work with a copy.
    on_fail :       "warn" | "ignore" | "raise"
                    What to do if points failed to xform.

    Returns
    -------
    xformed data
                    Returns same data type as input. Coordinates are returned
                    in pixel (at 4x4x40 nm).

    """
    return _flycon(x,
                   dataset='flywire_v1_inverse',
                   coordinates=coordinates,
                   inplace=inplace,
                   on_fail=on_fail,
                   mip=mip)


def flywire_to_fafb14(x, coordinates=None, mip=2, inplace=False, on_fail='warn'):
    """Transform neurons/coordinates from flywire to FAFB V14.

    This uses a service hosted by Eric Perlman.

    Parameters
    ----------
    x :             CatmaidNeuron/List | np.ndarray (N, 3)
                    Data to transform.
    mip :           int
                    Resolution of mapping. Lower = more precise but much slower.
    coordinates :   None | "nm" | "pixel"
                    Units of the provided data in ``x``. If ``None`` will
                    assume that Neuron/List are in nanometers and everything
                    else is in pixel.
    inplace :       bool
                    If ``True`` will modify Neuron object(s) in place. If ``False``
                    work with a copy.
    on_fail :       "warn" | "ignore" | "raise"
                    What to do if points failed to xform.

    Returns
    -------
    xformed data
                    Returns same data type as input. Coordinates are returned in
                    nm.

    """
    if isinstance(coordinates, type(None)):
        if isinstance(x, (navis.BaseNeuron, navis.NeuronList)):
            coordinates = 'nm'
        else:
            coordinates = 'pixel'

    xf = _flycon(x,
                 dataset='flywire_v1',
                 coordinates=coordinates,
                 inplace=inplace,
                 on_fail=on_fail,
                 mip=mip)

    # _flycon always returns pixels - we have to convert to back to nanometers
    if isinstance(xf, navis.NeuronList):
        for n in xf:
            if isinstance(n, navis.TreeNeuron):
                n *= [4, 4, 40, 1]
            elif isinstance(n, navis.MeshNeuron):
                n *= [4, 4, 40]
            xf.units = 'nm'  # manually set the units
    elif isinstance(xf, navis.TreeNeuron):
        # The 4th value is the radius and that is assumed to not change
        xf *= [4, 4, 40, 1]
        xf.units = 'nm'  # manually set the units
    elif hasattr(xf, 'vertices'):
        xf.vertices *= [4, 4, 40]
    else:
        xf *= [4, 4, 40]

    return xf


def _flycon(x, dataset, base_url='https://spine.janelia.org/app/transform-service',
            coordinates='nm', mip=2, inplace=False, on_fail='warn'):
    """Transform neurons/coordinates between flywire and FAFB V14.

    This uses a service hosted by Eric Perlman.

    Parameters
    ----------
    x :             CatmaidNeuron/List | np.ndarray (N, 3)
                    Data to transform.
    dataset :       str
                    Dataset to use for transform. Currently available:

                     - 'flywire_v1'
                     - 'flywire_v1_inverse' (only mip 4)

    base_url :      str
                    URL for xform service.
    mip :           int
                    Resolution of mapping. Lower = more precise but much slower.
                    Currently only mip >= 2 available.
    coordinates :   "nm" | "pixel"
                    Units of the provided coordinates in ``x``.
    inplace :       bool
                    If ``True`` will modify Neuron object(s) in place. If ``False``
                    work with a copy.
    on_fail :       "warn" | "ignore" | "raise"
                    What to do if points failed to xform.

    Returns
    -------
    xformed data
                    Returns same data type as input.

    """
    if isinstance(x, navis.NeuronList):
        return x.__class__([_flycon(n,
                                    dataset=dataset,
                                    on_fail=on_fail,
                                    coordinates=coordinates,
                                    mip=mip,
                                    base_url=base_url,
                                    inplace=inplace) for n in x])
    elif isinstance(x, (navis.BaseNeuron, navis.Volume, tm.Trimesh)):
        if not inplace:
            x = x.copy()

        if isinstance(x, navis.TreeNeuron):
            x.nodes[['x', 'y', 'z']] = _flycon(x.nodes[['x', 'y', 'z']].values,
                                               dataset=dataset,
                                               on_fail=on_fail,
                                               coordinates=coordinates,
                                               mip=mip,
                                               base_url=base_url,
                                               inplace=inplace)
        elif isinstance(x, (navis.MeshNeuron, navis.Volume, tm.Trimesh)):
            x.vertices = _flycon(x.vertices,
                                 dataset=dataset,
                                 on_fail=on_fail,
                                 coordinates=coordinates,
                                 mip=mip,
                                 base_url=base_url,
                                 inplace=inplace)
        else:
            raise TypeError(f'Unable to convert neuron of type "{type(x)}"')

        if isinstance(x, navis.BaseNeuron) and x.has_connectors:
            x.connectors[['x', 'y', 'z']] = _flycon(x.connectors[['x', 'y', 'z']].values,
                                                    dataset=dataset,
                                                    on_fail=on_fail,
                                                    coordinates=coordinates,
                                                    mip=mip,
                                                    base_url=base_url,
                                                    inplace=inplace)

        return x

    # Make sure we are working on array
    x = np.asarray(x)

    if x.ndim != 2 or x.shape[1] != 3:
        raise ValueError(f'Expected coordinates of shape (N, 3), got {x.shape}')

    # This returns offsets along x and y axis
    offsets = utils.query_spine(x, dataset,
                                query='transform',
                                coordinates=coordinates,
                                mip=mip,
                                on_fail=on_fail)

    # We need to cast x to the same type as offsets -> likely float 64
    x = x.astype(offsets.dtype)

    # Transform points
    x[:, :2] += offsets

    return x
