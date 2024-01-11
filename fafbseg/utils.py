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

import os
import six
import requests
import collections

import numpy as np

from pathlib import Path
from tqdm.auto import tqdm
from functools import wraps
from concurrent import futures
from collections.abc import Iterable
from urllib.parse import urlparse, urlencode

use_pbars = True

CACHE_DIR = "~/.fafbseg/cache/"


def never_cache(function):
    """Decorate to prevent caching of server responses."""

    @wraps(function)
    def wrapper(*args, **kwargs):
        # Find CATMAID instances
        instances = [v for k, v in kwargs.items() if "_instance" in k]

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


def make_url(*args, **GET):
    """Generate URL.

    Parameters
    ----------
    *args
                Will be turned into the URL. For example::

                    >>> make_url('http://my-server.com', 'skeleton', 'list')
                    'http://my-server.com/skeleton/list'

    **GET
                Keyword arguments are assumed to be GET request queries
                and will be encoded in the url. For example::

                    >>> make_url('http://my-server.com', 'skeleton', node_gt=100)
                    'http://my-server.com/skeleton?node_gt=100'

    Returns
    -------
    url :       str

    """
    # Generate the URL
    url = args[0]
    for arg in args[1:]:
        arg_str = str(arg)
        joiner = '' if url.endswith('/') else '/'
        relative = arg_str[1:] if arg_str.startswith('/') else arg_str
        url = requests.compat.urljoin(url + joiner, relative)
    if GET:
        url += '?{}'.format(urlencode(GET))
    return url


def make_iterable(x, force_type=None) -> np.ndarray:
    """Force input into a numpy array.

    For dicts, keys will be turned into array.
    """
    if not isinstance(x, Iterable) or isinstance(x, six.string_types):
        x = [x]

    if isinstance(x, (dict, set)):
        x = list(x)

    return np.asarray(x, dtype=force_type)


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
        with tqdm(total=len(self._chunk_map),
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

        return points, data.flatten()


def download_cache_file(url, filename=None, force_reload=False, verbose=True):
    """Load file from URL and cache locally.

    Parameters
    ----------
    url :           str
                    URL to file.
    filename :      str, optional
                    Filename to save to. If not explicitly provided will guess
                    filename from url.
    force_reload :  bool
                    If True, will force downloading file - even if it already
                    exists in the local cache.
    verbose :       bool

    Returns
    -------
    path :          pathlib.Path
                    Path to the locally cached file.

    """
    if not isinstance(url, str):
        raise TypeError(f"Expected `url` of type str, got {type(url)}")

    # Make sure the cache dir exists
    cache_dir = Path(CACHE_DIR).expanduser().absolute()
    if not cache_dir.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)

    if filename:
        fp = cache_dir / filename
    else:
        fp = cache_dir / Path(url).name

    if not fp.exists() or force_reload:
        if verbose and not os.environ.get("FAFBSEG_TESTING", False):
            print(
                f"Caching {fp.name} from {urlparse(url).netloc}... ", end="", flush=True
            )
        r = requests.get(url, allow_redirects=True)
        r.raise_for_status()
        content_type = r.headers.get("content-type", "").lower()
        is_text = "text" in content_type or "html" in content_type
        with open(fp, mode="w" if is_text else "w") as f:
            f.write(r.content.decode())
        if verbose and not os.environ.get("FAFBSEG_TESTING", False):
            print("Done.")

    return fp