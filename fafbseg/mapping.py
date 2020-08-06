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

import pymaid
import requests
import warnings

import numpy as np

from . import utils
use_pbars = utils.use_pbars

try:
    import msgpack
except ImportError:
    msgpack = None


def xform_flywire_fafb14(x, coordinates='nm', mip=2,
                         base_url='https://spine.janelia.org/app/flyconv',
                         inplace=False, on_fail='warn'):
    """Transform neurons/coordinates from flywire to FAFB V14.

    This uses a service hosted by Eric Perlman.

    Parameters
    ----------
    x :             CatmaidNeuron/List | np.ndarray (N, 3)
                    Data to transform.
    mip :           int
                    Resolution of mapping. Lower = more precise but much slower.
    coordinates :   "nm" | "pixel"
                    Units of the provided coordinates in ``x``.
    base_url :      str
                    URL for xform service.
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
    assert on_fail in ['warn', 'raise', 'ignore']
    assert coordinates in ['nm', 'pixel']
    assert isinstance(mip, (int, np.int))
    assert mip > 0

    if isinstance(x, pymaid.CatmaidNeuronList):
        return pymaid.CatmaidNeuronList([xform_flywire_fafb14(n,
                                                              on_fail=on_fail,
                                                              coordinates=coordinates,
                                                              mip=mip,
                                                              base_url=base_url,
                                                              inplace=inplace) for n in x])
    elif isinstance(x, pymaid.CatmaidNeuron):
        if not inplace:
            x = x.copy()

        x.nodes[['x', 'y', 'z']] = xform_flywire_fafb14(x.nodes[['x', 'y', 'z']].values,
                                                        on_fail=on_fail,
                                                        coordinates=coordinates,
                                                        mip=mip,
                                                        base_url=base_url,
                                                        inplace=inplace)

        if hasattr(x, 'connectors') and not x.connectors.empty:
            x.connectors[['x', 'y', 'z']] = xform_flywire_fafb14(x.connectors[['x', 'y', 'z']].values,
                                                                 on_fail=on_fail,
                                                                 coordinates=coordinates,
                                                                 mip=mip,
                                                                 base_url=base_url,
                                                                 inplace=inplace)

        return x

    # At this point we are expecting a numpy array
    if not isinstance(x, np.ndarray):
        x = np.array(x)

    # Make sure data is now in the correct format
    if not x.ndim == 2:
        raise TypeError('Expected 2d array, got {}'.format(x.ndim))
    if not x.shape[1] == 3:
        raise TypeError('Expected (N, 3) array, got {}'.format(x.shape))

    # We need to convert pixel coordinates
    if coordinates == 'nm':
        x = (x / [4, 4, 40]).astype(int)

    # Convert pixels to mip
    # x = (x / 2 ** mip).astype(int)

    # Generate URL
    url = f'{base_url}/dataset/flywire_v1/s/{mip}/values_array'

    # Generate payload
    payload = {'x': x[:, 0].tolist(),
               'y': x[:, 1].tolist(),
               'z': x[:, 2].tolist()}

    if msgpack:
        headers = {'Content-type': 'application/msgpack'}
        resp = requests.post(url,
                             data=msgpack.packb(payload),
                             headers=headers)

        # Check for errors
        resp.raise_for_status()
        data = msgpack.unpackb(resp.content)
    else:
        resp = requests.post(url,
                             json=payload,
                             headers={'Content-Type': 'application/json'})

        # Check for errors
        resp.raise_for_status()
        data = resp.json()

    if 'error' in data:
        raise ValueError('Server returned error: {}'.format(str(data)))

    # Parse data: service returns a list of dictionaries containing absolute
    # coordinates x/y/z and offsets dx/dy/dz
    coords = np.array([data['x'], data['y'], data['z']])

    # If mapping failed will contain NaNs
    is_nan = np.any(np.isnan(coords), axis=1)
    if np.any(is_nan):
        msg = '{} points failed to transform.'.format(is_nan.sum())
        if on_fail == 'warn':
            warnings.warn(msg)
        elif on_fail == 'raise':
            raise ValueError(msg)

    return coords
