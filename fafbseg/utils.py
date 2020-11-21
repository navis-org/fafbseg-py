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
"""Collection of utility functions."""

import navis
import requests
import warnings

import numpy as np
import trimesh as tm

from functools import wraps
from urllib.parse import urlparse

use_pbars = True

SERVICE_URL = 'https://spine.janelia.org/app/transform-service'


class OnDemandDict(dict):
    """Initialized with a just a URL.

    Will fetch data from given URL on first request.

    """

    def __init__(self, url):
        """Initialize with just a URL."""
        self.url = url
        self.fetched = False
        super().__init__()

    def __getitem__(self, key):
        if not self.fetched:
            self.update_from_url()
        return super().__getitem__(key)

    def __contains__(self, key):
        if not self.fetched:
            self.update_from_url()
        return super().__contains__(key)

    def get(self, *args, **kwargs):
        if not self.fetched:
            self.update_from_url()
        return super().get(*args, **kwargs)

    def update_from_url(self):
        """Update content from URL."""
        resp = requests.get(self.url)
        resp.raise_for_status()
        self.update(resp.json())
        self.fetched = True


SERVICE_INFO = OnDemandDict(f'{SERVICE_URL}/info')


def never_cache(function):
    """Decorate to prevent caching of server responses."""
    @wraps(function)
    def wrapper(*args, **kwargs):
        # Find CATMAID instances
        instances = [v for k, v in kwargs.items() if '_instance' in k]

        # Keep track of old caching settings
        old_values = [i.caching for i in instances]
        # Set caching to False
        for rm in instances:
            rm.caching = False
        try:
            # Execute function
            res = function(*args, **kwargs)
        except BaseException:
            raise
        finally:
            # Set caching to old value
            for rm, old in zip(instances, old_values):
                rm.caching = old
        # Return result
        return res
    return wrapper


def is_url(x):
    """Check if URL is valid."""
    try:
        result = urlparse(x)
        return all([result.scheme, result.netloc, result.path])
    except BaseException:
        return False


def spine_datasets():
    """Fetch datasets available on spine's transform service.

    Returns
    -------
    dict

    """
    resp = requests.get(f'{SERVICE_URL}/info')
    resp.raise_for_status()
    return resp.json()


def query_spine(x, dataset, query, coordinates='nm', mip=2,
               limit_request=10e9, on_fail='warn'):
    """Fetch data via the transform or dataset query service on spine.

    Parameters
    ----------
    x :             np.ndarray (N, 3) | Neuron/List | mesh
                    Data to transform.
    dataset :       str
                    Dataset to use for transform. Currently available:

                     - 'flywire_v1' (transformation)
                     - 'flywire_v1_inverse' (transformation)
                     - 'fanc_v4_to_v3' (transformation)
                     - 'fafb-ffn1-20200412' (query)
                     - 'fafb-ffn1-20200412-gcs' (query)
                     - 'fafb-ffn1-20200412-zarr' (query)
                     - 'flywire_190410' (query)

                    See also ``spine_datasets()`` for up-to-date info.
    query :         "query" | "transform"
                    Whether we "query" segmentation IDs at given location(s) or
                    we want to "transform" locations into a different space.
    mip :           int
                    Resolution of mapping. Lower = more precise but much slower.
    coordinates :   "nm" | "pixel"
                    Units of the provided coordinates in ``x``.
    on_fail :       "warn" | "ignore" | "raise"
                    What to do if points failed to xform.

    Returns
    -------
    response :      np.ndarray
                    (N, 2) of dx and dy coordinates for "transform"
                    (N, ) of IDs for "query"

    """
    # Hard-coded data types for now. From:
    # https://github.com/bocklab/transform_service/blob/master/app/config.py
    dtypes = {
              'test': np.float32,
              'flywire_v1': np.float32,
              'flywire_v1_inverse': np.float32,
              'fanc_v4_to_v3': np.float32,
              'fafb-ffn1-20200412': np.uint64,
              'fafb-ffn1-20200412-gcs': np.uint64,
              'fafb-ffn1-20200412-zarr': np.uint64,
              'flywire_190410': np.uint64
              }

    assert on_fail in ['warn', 'raise', 'ignore']
    assert query in ['query', 'transform']
    assert coordinates in ['nm', 'nanometers', 'nanometer', 'pixel', 'pxl', 'pixels']
    assert isinstance(mip, (int, np.int))
    assert mip >= 0

    if dataset not in SERVICE_INFO:
        ds_str = ", ".join(SERVICE_INFO.keys())
        raise ValueError(f'"{dataset}" not among listed datasets: {ds_str}')

    if mip not in SERVICE_INFO[dataset]['scales']:
        raise ValueError(f'mip {mip} not available for dataset "{dataset}". '
                         f'Available scales: {SERVICE_INFO[dataset]["scales"]}')

    if isinstance(x, (navis.NeuronList, navis.TreeNeuron)):
        x = x.nodes[['x', 'y', 'z']].values
    elif isinstance(x, (navis.Volume, navis.MeshNeuron, tm.Trimesh)):
        x = np.asarray(x.vertices)

    # At this point we are expecting a numpy array
    x = np.asarray(x)

    # Make sure data is now in the correct format
    if not x.ndim == 2:
        raise TypeError('Expected 2d array, got {}'.format(x.ndim))
    if not x.shape[1] == 3:
        raise TypeError('Expected (N, 3) array, got {}'.format(x.shape))

    # We need to convert to pixel coordinates
    # Note that we are rounding here to get to pixels
    # This will have the most impact on the Z section
    if coordinates in ['nm', 'nanometers', 'nanometer']:
        x = np.round(x / [4, 4, 40]).astype(int)

    # Generate URL
    url = f'{SERVICE_URL}/{query}/dataset/{dataset}/s/{mip}/values_binary/format/array_float_Nx3'

    # Make sure we don't exceed the maximum size for each request
    stack = []
    limit_request = int(limit_request)
    for ix in np.arange(0, x.shape[0], limit_request):
        this_x = x[ix: ix + limit_request]

        # Make request
        resp = requests.post(url,
                             data=this_x.astype(np.single).tobytes(order='C'))

        # Check for errors
        resp.raise_for_status()

        # Extract data
        data = np.frombuffer(resp.content, dtype=dtypes[dataset])
        if query == 'transform':
            data = data.reshape(x.shape[0], 2)
        stack.append(data)

    stack = np.concatenate(stack, axis=0)

    # If mapping failed will contain NaNs
    if on_fail != 'ignore':
        is_nan = np.any(np.isnan(stack),
                        axis=1 if stack.ndim == 2 else 0)
        if np.any(is_nan):
            msg = f'{is_nan.sum()} points failed to transform.'
            if on_fail == 'warn':
                warnings.warn(msg)
            elif on_fail == 'raise':
                raise ValueError(msg)

    return stack
