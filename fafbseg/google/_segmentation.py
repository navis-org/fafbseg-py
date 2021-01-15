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

"""DEPRECATED module."""

import cloudvolume
import collections
import numpy as np
import pandas as pd
import os
import requests
import tqdm

from concurrent import futures

from .. import utils
from .. import spine

CVtype = cloudvolume.frontends.precomputed.CloudVolumePrecomputed


__all__ = ['locs_to_segments']


class GSPointLoader(object):
    """Build up a list of points, then load them batched by storage chunk.

    This code is based on an implementation by
    `Peter Li<https://gist.github.com/chinasaur/5429ef3e0a60aa7a1c38801b0cbfe9bb>_.
    """

    def __init__(self, cloud_volume):
        """Initialize with zero points.

        See add_points to queue some.

        Parameters
        ----------
        cloud_volume :  cloudvolume.CloudVolume

        """
        if not isinstance(cloud_volume, CVtype):
            raise TypeError('Expected CloudVolume, got "{}"'.format(type(cloud_volume)))

        self._volume = cloud_volume
        self._chunk_map = collections.defaultdict(set)
        self._points = None

    def add_points(self, points):
        """Add more points to be loaded.

        Parameters
        ----------
        points:     iterable of XYZ iterables
                    E.g. Nx3 ndarray.  Assumed to be in absolute units relative
                    to volume.scale['resolution'].

        """
        points = np.asarray(points)

        if isinstance(self._points, type(None)):
            self._points = points
        else:
            self._points = np.concat(self._points, points)

        resolution = np.array(self._volume.scale['resolution'])
        chunk_size = np.array(self._volume.scale['chunk_sizes'])
        chunk_starts = (points // resolution).astype(int) // chunk_size * chunk_size
        for point, chunk_start in zip(points, chunk_starts):
            self._chunk_map[tuple(chunk_start)].add(tuple(point))

    def _load_chunk(self, chunk_start, chunk_end):
        # (No validation that this is a valid chunk_start.)
        return self._volume[chunk_start[0]:chunk_end[0],
                            chunk_start[1]:chunk_end[1],
                            chunk_start[2]:chunk_end[2]]

    def _load_points(self, chunk_map_key):
        chunk_start = np.array(chunk_map_key)
        points = np.array(list(self._chunk_map[chunk_map_key]))

        resolution = np.array(self._volume.scale['resolution'])
        indices = (points // resolution).astype(int) - chunk_start

        # We don't really need to load the whole chunk here:
        # Instead, we subset the chunk to the part that contains our points
        # This should at the very least save memory
        mn, mx = indices.min(axis=0), indices.max(axis=0)

        chunk_end = chunk_start + mx + 1
        chunk_start += mn
        indices -= mn

        chunk = self._load_chunk(chunk_start, chunk_end)
        return points, chunk[indices[:, 0], indices[:, 1], indices[:, 2]]

    def load_all(self, max_workers=4, return_sorted=True, progress=True):
        """Load all points in current list, batching by storage chunk.

        Parameters
        ----------
        max_workers :   int, optional
                        The max number of workers for parallel chunk requests.
        return_sorted : bool, optional
                        If True, will order the returned data to match the order
                        of the points as they were added.
        progress :      bool, optional
                        Whether to show progress bar.

        Returns
        -------
        points :        np.ndarray
        data :          np.ndarray
                        Parallel Numpy arrays of the requested points from all
                        cumulative calls to add_points, and the corresponding
                        data loaded from volume.

        """
        progress_state = self._volume.progress
        self._volume.progress = False
        with tqdm.tqdm(total=len(self._chunk_map),
                       desc='Segmentation IDs',
                       leave=False,
                       disable=not progress) as pbar:
            with futures.ProcessPoolExecutor(max_workers=max_workers) as ex:
                point_futures = [ex.submit(self._load_points, k) for k in self._chunk_map]
                for f in futures.as_completed(point_futures):
                    pbar.update(1)
        self._volume.progress = progress_state

        results = [f.result() for f in point_futures]

        if return_sorted:
            points_dict = dict(zip([tuple(p) for result in results for p in result[0]],
                                   [i for result in results for i in result[1]]))

            data = np.array([points_dict[tuple(p)] for p in self._points])
            points = self._points
        else:
            points = np.concatenate([result[0] for result in results])
            data = np.concatenate([result[1] for result in results])

        return points, data


def _get_seg_ids_gs(points, volume, max_workers=4, progress=True):
    """DEPCREATED. Fetch segment IDs using CloudVolume hosted on Google Storage.

    This is the default option as it does not require any credentials. Downside:
    it's slow!

    Parameters
    ----------
    points :            list-like
                        x/y/z coordinates in absolute units.
    volume :            cloudvolume.CloudVolume
                        The CloudVolume to query.
    max_workers :       int, optional
                        Maximal number of parallel queries.
    progress :          bool, optional
                        If False, will not show progress bar.

    Returns
    -------
    list :              List of segmentation IDs at given locations.

    """
    pl = GSPointLoader(volume)
    pl.add_points(points)

    points, data = pl.load_all(max_workers=max_workers,
                               progress=progress,
                               return_sorted=True)

    return data


def _get_seg_ids_url(locs, url=None, voxel_conversion=[8, 8, 40],
                     payload_format='locations', chunk_size=10e3, progress=True,
                     **kwargs):
    """DEPCREATED. Fetch segment IDs at given locations via a URL.

    Use this is you are hosting the segmentation data yourself on a remote
    server. The server must accept POST requests to the given URL with
    a list of x/y/z coordinates as payload.

    Parameters
    ----------
    locs :              list-like
                        x/y/z coordinates in absolute units.
    url :               str, optional
                        Specify the URL to query for segment IDs.
    voxel_conversion :  list-like, optional
                        Size of each voxel. This is used to convert from
                        absolute units to voxel coordinates.
    chunk_size :        int, optional
                        Use this to limit the number of locations per query.
    progress :          bool, optional
                        If False, will not show progress bar.
    **kwargs
                        Keyword arguments passed to ``requests.post``.

    Returns
    -------
    list :              List of segmentation IDs.

    """
    # TODO:
    # - group locs such that each query ideally contains only close-by coordinates
    assert payload_format in ('locations', 'columns')

    if not url:
        url = os.environ.get('SEG_ID_URL')

    if not url:
        raise ValueError('Must provide valid URL to fetch segment IDs')

    # Make sure locations are numpy array
    locs = np.asarray(locs)

    # Make sure voxel_conversion is array
    voxel_conversion = np.array(voxel_conversion)

    # Bring locations into voxel space
    locs = (locs / voxel_conversion).astype(int)

    # Turn into pandas DataFrame
    locs = pd.DataFrame(locs, columns=['x', 'y', 'z'])

    # Guess the chunk within the EM volume that each locations belongs to
    cs = np.array([128, 128, 32])
    locs['cs'] = (locs // cs * cs).apply(tuple, axis=1)

    # Sort by chunk start - the index will keep track of the original order
    locs.sort_values('cs', inplace=True)

    seg_ids = []
    with tqdm.tqdm(total=len(locs),
                   desc='Segment IDs',
                   leave=False,
                   disable=not progress) as pbar:
        for i in range(0, len(locs), int(chunk_size)):
            chunk = locs.iloc[i: i + int(chunk_size)][['x', 'y', 'z']].values

            if payload_format == 'locations':
                payload = {'locations': chunk.tolist()}
            elif payload_format == 'columns':
                payload = {'x': chunk[:, 0].tolist(),
                           'y': chunk[:, 1].tolist(),
                           'z': chunk[:, 2].tolist()}

            resp = requests.post(url, json=payload, **kwargs)
            resp.raise_for_status()

            if 'error' in resp.json():
                raise BaseException('Error fetching data: {}'.format(resp.json()))

            data = resp.json()

            if isinstance(data, list):
                seg_ids = np.append(seg_ids, data)
            elif isinstance(data, dict) and 'values' in data:
                seg_ids = np.append(seg_ids, data['values'])
            else:
                raise ValueError(f'Unable to parse response: {data}')

            pbar.update(len(chunk))

    locs['seg_ids'] = np.array(seg_ids).flatten()

    # Get back to original order and return
    return locs.reset_index(drop=False).sort_values('index').seg_ids.values


def use_google_storage(volume, max_workers=8, progress=True, **kwargs):
    """DEPCREATED. Use Google Storage via CloudVolume for segmentation IDs.

    Parameters
    ----------
    volume :        str | CloudVolume
                    Name or URL of CloudVolume to use to fetch segmentation IDs.
    max_workers :   int, optional
                    Maximal number of parallel queries.
    progress :      bool, optional
                    If False, will not show progress bar.
    **kwargs
                    Keyword arguments passed on to ``cloudvolume.CloudVolume``.


    Returns
    -------
    None

    Examples
    --------
    # Segmentation for FAFB autoseg V3
    >>> fafbseg.use_google_storage("https://storage.googleapis.com/fafb-ffn1-20190805/segmentation")

    # Also works with just the ID
    >>> fafbseg.use_google_storage("fafb-ffn1-20190805")

    See Also
    --------
    :func:`~fafbseg.use_brainmaps`
                        Use this if you have access to the brainmaps API.
    :func:`~fafbseg.use_remote_service`
                        Use this is if you are hosting your own solution.
    :func:`~fafbseg.use_local_data`
                        Use this is if you have a local copy of the segmentation
                        data.

    """
    global _get_seg_ids

    if 'CloudVolume' not in str(type(volume)):
        # Set and update defaults from kwargs
        defaults = dict(cache=True,
                        mip=0,
                        progress=False)
        defaults.update(kwargs)

        if 'http' in volume:
            url = volume
        else:
            url = 'https://storage.googleapis.com/{}/segmentation'.format(volume)

        volume = cloudvolume.CloudVolume(url, **defaults)

    _get_seg_ids = lambda x: _get_seg_ids_gs(x, volume,
                                             max_workers=max_workers,
                                             progress=progress)
    print('Using Google CloudStorage to retrieve segmentation IDs.')


def use_local_data(path, progress=True, **kwargs):
    """DEPCREATED. Fetch segment IDs using a local copy of the segmentation data.

    Parameters
    ----------
    path :          str
                    Path to the local copy. Must point to the directory with
                    the ``info`` file.
    progress :      bool, optional
                    If False, will not show progress bar.
    **kwargs
                    Keyword arguments passed on to ``cloudvolume.CloudVolume``.


    Returns
    -------
    None

    Examples
    --------
    >>> fafbseg.use_local_data("/Volumes/SSD/segmentation")

    See Also
    --------
    :func:`~fafbseg.use_brainmaps`
                        Use this if you have access to the brainmaps API.
    :func:`~fafbseg.use_remote_service`
                        Use this is if you are hosting your own solution.
    :func:`~fafbseg.use_google_storage`
                        Use this to access via Google Cloud storage.

    """
    global _get_seg_ids

    # Set and update defaults from kwargs
    defaults = dict(cache=True,
                    mip=0,
                    progress=False)
    defaults.update(kwargs)

    if not path.startswith('file://'):
        path = 'file://' + path

    volume = cloudvolume.CloudVolume(path, **defaults)
    _get_seg_ids = lambda x: _get_seg_ids_gs(x, volume,
                                             max_workers=1,
                                             progress=progress)
    print('Using local segmentation data to retrieve segmentation IDs.')


def use_remote_service(url=None, voxel_conversion=[8, 8, 40],
                       payload_format='locations', chunk_size=10e3,
                       **kwargs):
    """DEPCREATED. Fetch segment IDs at given locations using a custom web service.

    Use this is you are hosting the segmentation data yourself on a remote
    server. The server must accept POST requests to the given URL with
    a list of x/y/z coordinates as payload.

    Parameters
    ----------
    url :               str, optional
                        Specify the URL to query for segment IDs. If not
                        provided will look for ``SEG_ID_URL`` environment
                        variable.
    voxel_conversion :  list-like, optional
                        Size of each voxel. This is used to convert from
                        absolute units to voxel coordinates.
    payload_format :    "locations" | "columns"
                        Format for the query JSON payload::

                         'locations': {'locations': [[x1, y1, z1], ..]}
                         'columns': {'x': [x1, ..], 'y': [y1, ..], 'z': [z1, ..]}

    chunk_size :        int, optional
                        Use this to limit the number of locations per query.
    **kwargs
                        Keyword arguments passed to ``requests.post``.

    Returns
    -------
    None

    Examples
    --------
    Set url in Python:

    >>> fafbseg.use_remote_service('https://my-server.com/seg/values')

    Alternatively, set an environment variable:

    $ EXPORT SEG_ID_URL="https://my-server.com/seg/values"

    >>> fafbseg.use_remote_service()

    See Also
    --------
    :func:`~fafbseg.use_brainmaps`
                        Use this if you have access to the brainmaps API.
    :func:`~fafbseg.use_google_storage`
                        This uses the segmentation data hosted on Google Storage
                        and does not require any special permissions.
    :func:`~fafbseg.use_local_data`
                        Use this is if you have a local copy of the segmentation
                        data.

    """
    global _get_seg_ids

    if not url:
        url = os.environ.get('SEG_ID_URL')

    if not utils.is_url(url):
        raise ValueError("Invalid URL. Must provide valid URL.")

    _get_seg_ids = lambda x: _get_seg_ids_url(x, url,
                                              voxel_conversion=voxel_conversion,
                                              chunk_size=chunk_size,
                                              payload_format=payload_format,
                                              **kwargs)
    print('Using web-hosted solution to retrieve segmentation IDs.')


def use_brainmaps(volume_id, client_secret=None, max_threads=10):
    """DEPCREATED. Fetch segment IDs at given locations using the brainmaps API.

    This requires you to have brainmaps API access and the brainmappy Python
    package installed. See `brainmappy <https://github.com/schlegelp/brainmappy>`_
    for details on how to install it and acquire credentials.

    Parameters
    ----------
    volume_id :         str, optional
                        ID of volume to be queried against.
    client_secret :     str, optional
                        If you are authenticating for the first time, you need
                        to provide a `client_secret.json`. Not necessary on
                        subsequent logins.
    max_threads :       int, optional
                        Max number of threads to be used for querying against
                        brainmaps API.

    Returns
    -------
    None

    See Also
    --------
    :func:`~fafbseg.use_remote_service`
                        Use this is if you are hosting your own solution.
    :func:`~fafbseg.use_google_storage`
                        This uses the segmentation data hosted on Google Storage
                        and does not require any special permissions.
    :func:`~fafbseg.use_local_data`
                        Use this is if you have a local copy of the segmentation
                        data.

    """
    global _get_seg_ids

    try:
        import brainmappy as bm
    except ImportError as e:
        raise ImportError('Must have brainmappy installed. See '
                          'https://github.com/schlegelp/brainmappy '
                          'on how to install.') from e

    session = bm.acquire_credentials(client_secret)

    _get_seg_ids = lambda x: bm.get_seg_at_location(x,
                                                    volume_id=volume_id,
                                                    max_threads=max_threads,
                                                    session=session
                                                    )
    print('Using brainmaps API to retrieve segmentation IDs.')


def locs_to_segments(locs, mip=0, dataset='fafb-ffn1-20200412',
                     coordinates='voxel'):
    """Retrieve Google segmentation IDs at given location(s).

    Uses Eric Perlman's and Davi Bock's service on spine.janelia.org

    Parameters
    ----------
    locs :          list-like
                    Array of x/y/z coordinates.
    mip :           int [0-9]
                    Scale to query. Lower mip = more precise but slower;
                    higher mip = faster but less precise (small segments
                    might not show at all at high mips).
    dataset :       str
                    Currently, the only available dataset is
                    "fafb-ffn1-20200412", the most recent segmentation by Google.
    coordinates :   "voxel" | "nm"
                    Units in which your coordinates are in. "voxel" is assumed
                    to be 4x4x40 (x/y/z) nanometers.

    Returns
    -------
    numpy.array
                List of segmentation IDs in the same order as ``locs``. Invalid
                locations will be returned with ID 0.

    """
    return spine.transform.get_segids(locs, segmentation=dataset,
                                      coordinates=coordinates, mip=mip)


def __locs_to_segments(locs, progress=True):
    """DEPCREATED. Retrieve segmentation IDs at given location(s).

    On startup you must use one of the ``fafbseg.use_...`` functions (see
    below) to set a source for the segmentation data.

    Parameters
    ----------
    locs :      list-like
                Array of x/y/z coordinates in NM.

    Returns
    -------
    np.array
                List of segmentation IDs in the same order as ``locs``.
                Locations without segmentation will typically return with ID 0.

    See Also
    --------
    :func:`~fafbseg.use_remote_service`
                        Use this is if you are hosting your own solution.
    :func:`~fafbseg.use_google_storage`
                        This uses the segmentation data hosted on Google Storage
                        and does not require any special permissions.
    :func:`~fafbseg.use_local_data`
                        Use this is if you have a local copy of the segmentation
                        data.
    :func:`~fafbseg.use_brainmaps`
                        Use this if you have access to the brainmaps API.

    """
    locs = np.asarray(locs)
    if locs.ndim == 1 and len(locs) == 3:
        locs = locs.reshape((1, 3))
    elif locs.ndim != 2 or locs.shape[1] != 3:
        raise ValueError('Expected x/y/z coordinates as array of shape (N, 3)')

    ids = _get_seg_ids(locs)

    # Do some clean-up
    # First, make sure this is an array
    ids = np.asarray(ids)

    # Do not remove the flatten here -> some service return nested lists/arrays
    ids = ids.flatten()

    # Replace NaNs/None with 0
    ids[np.isin(ids, [None, 'NaN', 'None', 'nan'])] = 0

    # Coerce to integers
    ids = ids.astype(int, copy=False)

    return ids


def _warn_setup(*args, **kwargs):
    """Tell user to set up connection."""
    raise BaseException('Please use fafbseg.use_google_storage, '
                        'fafbseg.use_brainmaps, fafbseg.use_remote_service '
                        'or fafbseg.use_local_data '
                        'to set the way you want to fetch segmentation IDs.')


# On import access to segmentation is not set -> this function will warn
_get_seg_ids = _warn_setup
