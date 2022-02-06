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

from navis.transforms.base import BaseTransform, AliasTransform
from navis.transforms.affine import AffineTransform

from .. import utils, spine
use_pbars = utils.use_pbars

__all__ = ['fafb14_to_flywire', 'flywire_to_fafb14', 'register_transforms']


class SpineTransform(BaseTransform):
    """Transform data using the spine web service.

    Parameters
    ----------
    fw_dataset :    str
                    API endpoint for forward transform.
    inv_dataset :   str
                    API endpoint for forward transform.
    direction :     'forward' | 'inverse'
                    Direction of transform.
    mip :           int
                    Resolution to use for forward mip. 0 = highest resolution.
                    Negative values start counting from the highest possible
                    resolution: -1 = highest, -2 = second highest, etc.
    coordinates :   "voxels" | "nanometers"
                    Whether coordinates of points are expected to be in raw
                    voxel or in nanometers.
    on_fail :       "ignore" | "warn" | "raise"
                    What to do if coordinates fail to transform.

    """

    def __init__(self,
                 fw_dataset: str,
                 inv_dataset:  str,
                 direction: str = 'forward',
                 mip: int = -1,
                 coordinates: str = 'voxels',
                 on_fail: str = 'warn'):
        """Initialize."""
        assert isinstance(fw_dataset, str)
        assert isinstance(inv_dataset, str)
        assert direction in ('forward', 'inverse')
        assert isinstance(mip, (int, np.int))
        assert coordinates in ('voxels', 'nanometers')
        assert on_fail in ('warn', 'ignore', 'raise')

        self.fw_dataset = fw_dataset
        self.inv_dataset = inv_dataset
        self.mip = mip
        self.coordinates = coordinates
        self.on_fail = on_fail

        self.direction = direction

    def __neg__(self):
        """Switch directions."""
        # Invert direction
        new_direction = {'forward': 'inverse',
                         'inverse': 'forward'}[self.direction]

        return SpineTransform(self.fw_dataset,
                              self.inv_dataset,
                              direction=new_direction,
                              mip=self.mip,
                              coordinates=self.coordinates,
                              on_fail=self.on_fail)

    def copy(self):
        """Return copy."""
        return SpineTransform(self.fw_dataset,
                              self.inv_dataset,
                              direction=self.direction,
                              mip=self.mip,
                              coordinates=self.coordinates,
                              on_fail=self.on_fail)

    def xform(self, points: np.ndarray) -> np.ndarray:
        """Transform points.

        Parameters
        ----------
        points :    (N, 3) np.ndarray
                    x/y/z coordinates to transform.

        Returns
        -------
        xf :        (N, 3) np.ndarray
                    Transformed coordinates.

        """
        if self.direction == 'forward':
            dataset = self.fw_dataset
        else:
            dataset = self.inv_dataset

        # This returns offsets along x and y axis
        offsets = spine.transform.get_offsets(points,
                                              transform=dataset,
                                              coordinates=self.coordinates,
                                              mip=self.mip,
                                              on_fail=self.on_fail)

        # We need to cast x to the same type as offsets -> likely float 64
        # This also makes a copy - do not change that!
        xf = points.astype(offsets.dtype)

        # Apply offsets
        xf[:, :2] += offsets

        return xf


def register_transforms():
    """Register spine transforms with navis."""
    # FAFB14 <-> FAFB14.1 (flywire) - note both of these are in voxels
    tr = SpineTransform(fw_dataset='flywire_v1',
                        inv_dataset='flywire_v1_inverse',
                        coordinates='voxels',
                        mip=-1)
    navis.transforms.registry.register_transform(tr,
                                                 source='FLYWIREraw',
                                                 target='FAFB14raw',
                                                 transform_type='bridging')

    # Add transform between FAFB14 (nm) and FAFB14raw (4x4x40nm voxels)
    # and between FLYWIRE (nm) and FLYWIREraw (4x4x40nm voxels)
    nm_to_voxel = AffineTransform(np.diag([4, 4, 40, 1]))
    navis.transforms.registry.register_transform(transform=nm_to_voxel,
                                                 source='FAFB14raw',
                                                 target='FAFB14',
                                                 transform_type='bridging')
    navis.transforms.registry.register_transform(transform=nm_to_voxel,
                                                 source='FLYWIREraw',
                                                 target='FLYWIRE',
                                                 transform_type='bridging')

    # Add alias transform between FLYWIRE and FAFB14.1 (they are synonymous)
    navis.transforms.registry.register_transform(transform=AliasTransform(),
                                                 source='FLYWIREraw',
                                                 target='FAFB14.1raw',
                                                 transform_type='bridging')
    navis.transforms.registry.register_transform(transform=AliasTransform(),
                                                 source='FLYWIRE',
                                                 target='FAFB14.1',
                                                 transform_type='bridging')
    navis.transforms.registry.register_transform(transform=AliasTransform(),
                                                 source='FAFB',
                                                 target='FAFB14',
                                                 transform_type='bridging')


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
    coordinates :   "nm" | "voxel"
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
                    in the same coordinate space (voxels or nm) as the input.

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
    coordinates :   None | "nm" | "voxel"
                    Units of the provided data in ``x``. If ``None`` will
                    assume that Neuron/List are in nanometers and everything
                    else is in voxel.
    inplace :       bool
                    If ``True`` will modify Neuron object(s) in place. If ``False``
                    work with a copy.
    on_fail :       "warn" | "ignore" | "raise"
                    What to do if points failed to xform.

    Returns
    -------
    xformed data
                    Returns same data type and in the same coordinates space
                    (nm or voxel) as the input.

    """
    if isinstance(coordinates, type(None)):
        if isinstance(x, (navis.BaseNeuron, navis.NeuronList)):
            coordinates = 'nm'
        else:
            coordinates = 'voxel'

    xf = _flycon(x,
                 dataset='flywire_v1',
                 coordinates=coordinates,
                 inplace=inplace,
                 on_fail=on_fail,
                 mip=mip)

    return xf


def _flycon(x, dataset, base_url='https://services.itanna.io/app/transform-service',
            coordinates='nm', mip=2, inplace=False, on_fail='warn'):
    """DEPCREATED! Transform neurons/coordinates between flywire and FAFB V14.

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
    coordinates :   "nm" | "voxel"
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
    offsets = spine.transform.get_offsets(x, transform=dataset,
                                          coordinates=coordinates,
                                          mip=mip,
                                          on_fail=on_fail)

    # `offsets` will always be in voxels - if our data is in nanometers, we have
    #  to convert them
    if coordinates in ('nm', 'nanometers', 'nanometres'):
        offsets *= 4

    # We need to cast x to the same type as offsets -> likely float 64
    x = x.astype(offsets.dtype)

    # Transform points
    x[:, :2] += offsets

    return x
