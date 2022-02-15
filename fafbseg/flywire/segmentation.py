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

import pymaid
import navis
import requests
import textwrap
import time

import datetime as dt
import numpy as np
import pandas as pd

from concurrent import futures
from diskcache import Cache
from requests_futures.sessions import FuturesSession
from scipy import ndimage
from tqdm.auto import trange, tqdm

from .. import spine
from .. import xform

from .utils import (parse_volume, FLYWIRE_DATASETS, get_chunkedgraph_secret,
                    retry, get_cave_client)


__all__ = ['fetch_edit_history', 'fetch_leaderboard', 'locs_to_segments',
           'locs_to_supervoxels', 'skid_to_id', 'update_ids',
           'roots_to_supervoxels', 'supervoxels_to_roots',
           'neuron_to_segments', 'is_latest_root', 'is_valid_root',
           'is_valid_supervoxel']


def fetch_leaderboard(days=7, by_day=False, progress=True, max_threads=4):
    """Fetch leader board (# of edits).

    Parameters
    ----------
    day :           int
                    Number of days to go back.
    by_day :        bool
                    If True, will provide a day-by-day breakdown of # edits.
    progress :      bool
                    If True, show progress bar.
    max_threads :   int
                    Max number of parallel requests to server.

    Returns
    -------
    pandas.DataFrame

    Examples
    --------
    >>> from fafbseg import flywire
    >>> # Fetch leaderboard with edits per day
    >>> hist = flywire.fetch_leaderboard(by_day=True)
    >>> # Plot user actions over time
    >>> hist.T.plot()

    """
    assert isinstance(days, (int, np.int))
    assert days >= 0

    session = requests.Session()
    if not by_day:
        url = f'https://pyrdev.eyewire.org/flywire-leaderboard?days={days-1}'
        resp = session.get(url, params=None)
        resp.raise_for_status()
        return pd.DataFrame.from_records(resp.json()['entries']).set_index('name')

    future_session = FuturesSession(session=session, max_workers=max_threads)
    futures = []
    for i in range(0, days):
        url = f'https://pyrdev.eyewire.org/flywire-leaderboard?days={i}'
        futures.append(future_session.get(url, params=None))

    # Get the responses
    resp = [f.result() for f in navis.config.tqdm(futures,
                                                  desc='Fetching',
                                                  disable=not progress or len(futures) == 1,
                                                  leave=False)]

    df = None
    for i, r in enumerate(resp):
        date = dt.date.today() - dt.timedelta(days=i)
        r.raise_for_status()
        this_df = pd.DataFrame.from_records(r.json()['entries']).set_index('name')
        this_df.columns = [date]
        if isinstance(df, type(None)):
            df = this_df
        else:
            df = pd.merge(df, this_df, how='outer', left_index=True, right_index=True)

    # Make sure we don't have NAs
    df = df.fillna(0).astype(int)

    # This breaks it down into days
    if df.shape[1] > 1:
        df.iloc[:, 1:] = df.iloc[:, 1:].values - df.iloc[:, :-1].values

    # Reverse such that the right-most entry is the current date
    df = df.iloc[:, ::-1]
    return df.loc[df.sum(axis=1).sort_values(ascending=False).index]


def fetch_edit_history(x, dataset='production', progress=True, max_threads=4):
    """Fetch edit history for given neuron(s).

    Parameters
    ----------
    x :             int | list of int
                    Segmentation (root) ID(s).
    dataset :       str | CloudVolume
                    Against which FlyWire dataset to query::
                        - "production" (current production dataset, fly_v31)
                        - "sandbox" (i.e. fly_v26)
    progress :      bool
                    If True, show progress bar.
    max_threads :   int
                    Max number of parallel requests to server.

    Returns
    -------
    pandas.DataFrame

    Examples
    --------
    >>> from fafbseg import flywire
    >>> # Fetch edits
    >>> edits = flywire.fetch_edit_history(720575940621039145)
    >>> # Group by user
    >>> edits.groupby('user_name').size()
    user_name
    Claire McKellar    47
    Jay Gager           4
    Sandeep Kumar       1
    Sarah Morejohn      6
    dtype: int64

    """
    if not isinstance(x, (list, set, np.ndarray)):
        x = [x]

    session = requests.Session()
    future_session = FuturesSession(session=session, max_workers=max_threads)

    token = get_chunkedgraph_secret()
    session.headers['Authorization'] = f"Bearer {token}"

    futures = []
    for id in x:
        dataset = FLYWIRE_DATASETS.get(dataset, dataset)
        url = f'https://prod.flywire-daf.com/segmentation/api/v1/table/{dataset}/root/{id}/tabular_change_log'
        f = future_session.get(url, params=None)
        futures.append(f)

    # Get the responses
    resp = [f.result() for f in navis.config.tqdm(futures,
                                                  desc='Fetching',
                                                  disable=not progress or len(futures) == 1,
                                                  leave=False)]

    df = []
    for r, i in zip(resp, x):
        r.raise_for_status()
        this_df = pd.DataFrame(r.json())
        this_df['segment'] = i
        df.append(this_df)

    # Concat if any edits at all
    if any([not f.empty for f in df]):
        # Drop neurons without edits
        df = [f for f in df if not f.empty]
        df = pd.concat(df, axis=0, sort=True)
        df['timestamp'] = pd.to_datetime(df.timestamp, unit='ms')
    else:
        # Return the first empty data frame
        df = df[0]

    return df


def roots_to_supervoxels(x, use_cache=True, dataset='production', progress=True):
    """Get supervoxels making up given neurons.

    Parameters
    ----------
    x :             int | list of int
                    Segmentation (root) ID(s).
    use_cache :     bool
                    Whether to use disk cache to avoid repeated queries for the
                    same root. Cache is stored in `~/.fafbseg/`.
    dataset :       str | CloudVolume
                    Against which FlyWire dataset to query::
                        - "production" (current production dataset, fly_v31)
                        - "sandbox" (i.e. fly_v26)
    progress :      bool
                    If True, show progress bar.

    Returns
    -------
    dict
                    ``{root_id: [svoxel_id1, svoxelid2, ...], ...}``

    Examples
    --------
    >>> from fafbseg import flywire
    >>> flywire.roots_to_supervoxels(720575940619164912)[720575940619164912]
    array([78180637324085660, 78180706043394027, 78180706043400870, ...,
           78743587210799793, 78743587210799781, 78743587210818108],
          dtype=uint64)

    """
    # Make sure we are working with an array of integers
    x = navis.utils.make_iterable(x).astype(int, copy=False)

    if len(x) <= 1:
        progress = False

    # Get the volume
    vol = parse_volume(dataset)

    svoxels = {}
    # See what ewe can get from cache
    if use_cache:
        # Cache for root -> supervoxels
        # Grows to max 1Gb by default and persists across sessions
        with Cache(directory='~/.fafbseg/svoxel_cache/') as sv_cache:
            # See if we have any of these roots cached
            with sv_cache.transact():
                is_cached = np.isin(x, sv_cache)

            # Add supervoxels from cache if we have any
            if np.any(is_cached):
                # Get values from cache
                with sv_cache.transact():
                    svoxels.update({i: sv_cache[i] for i in x[is_cached]})

    # Get the supervoxels for the roots that are still missing
    # We need to convert keys to integer array because otherwise there is a
    # mismatch in types (int vs np.int?) which causes all root IDs to be in miss
    # -> I think that's because of the way disk cache works
    miss = x[~np.isin(x, np.array(list(svoxels.keys())).astype(int))]
    get_leaves = retry(vol.get_leaves)
    svoxels.update({i: get_leaves(i,
                                  bbox=vol.meta.bounds(0),
                                  mip=0) for i in navis.config.tqdm(miss,
                                                                    desc='Querying',
                                                                    disable=not progress,
                                                                    leave=False)})

    # Update cache
    if use_cache:
        with sv_cache.transact():
            for i in miss:
                sv_cache[i] = svoxels[i]

    return svoxels


def supervoxels_to_roots(x, timestamp=None, batch_size=10_000,
                         retry=True, progress=True, dataset='production'):
    """Get root(s) for given supervoxel(s).

    Parameters
    ----------
    x :             int | list of int
                    Supervoxel ID(s) to find the root(s) for.
    timestamp :     int | str | datetime
                    Get roots at given date (and time). Int must be unix
                    timestamp. String must be ISO 8601 - e.g. '2021-11-15'.
                    If timestamp is given, will ignore `use_cache`.
    batch_size :    int
                    Max number of supervoxel IDs per query. Reduce batch size if
                    you experience time outs.
    retry :         bool
                    Whether to retry if a batched query fails.
    dataset :       str | CloudVolume
                    Against which dataset to query::
                        - "production" (current production dataset, fly_v31)
                        - "sandbox" (i.e. fly_v26)
    progress :      bool
                    If True, show progress bar.

    Returns
    -------
    roots  :        numpy array
                    Roots corresponding to supervoxels in `x`.

    Examples
    --------
    >>> from fafbseg import flywire
    >>> flywire.supervoxels_to_roots(78321855915861142)

    """
    # Make sure we are working with an array of integers
    x = navis.utils.make_iterable(x)

    # Check if IDs are valid (zeros are fine because we filter for them later on)
    is_valid_supervoxel(x[(x != 0) & (x != '0')], raise_exc=True)

    x = x.astype(np.int64, copy=False)

    # Parse the volume
    vol = parse_volume(dataset)

    # Prepare results array
    roots = np.zeros(x.shape, dtype=np.int64)

    if isinstance(timestamp, np.datetime64):
        timestamp = str(timestamp)

    with tqdm(desc='Fetching roots',
              leave=False,
              total=len(x),
              disable=not progress or len(x) < batch_size) as pbar:

        for i in range(0, len(x), int(batch_size)):
            # This batch
            batch = x[i:i+batch_size]

            # get_roots() doesn't like to be asked for zeros - causes server error
            not_zero = batch != 0
            try:
                roots[i:i+batch_size][not_zero] = vol.get_roots(batch[not_zero],
                                                                timestamp=timestamp)
            except BaseException:
                if not retry:
                    raise
                time.sleep(1)
                roots[i:i+batch_size][not_zero] = vol.get_roots(batch[not_zero],
                                                                timestamp=timestamp)

            pbar.update(len(batch))

    return roots


def locs_to_supervoxels(locs, mip=2, coordinates='voxel'):
    """Retrieve FlyWire supervoxel IDs at given location(s).

    Use Eric Perlman's service on spine.

    Parameters
    ----------
    locs :          list-like | pandas.DataFrame
                    Array of x/y/z coordinates. If DataFrame must contain
                    'x', 'y', 'z' or 'fw.x', 'fw.y', 'fw.z' columns. If both
                    present, 'fw.' columns take precedence!
    mip :           int [2-8]
                    Scale to query. Lower mip = more precise but slower;
                    higher mip = faster but less precise (small supervoxels
                    might not show at all).
    coordinates :   "voxel" | "nm"
                    Units in which your coordinates are in. "voxel" is assumed
                    to be 4x4x40 (x/y/z) nanometers.

    Returns
    -------
    numpy.array
                List of segmentation IDs in the same order as ``locs``. Invalid
                locations will be returned with ID 0.

    Examples
    --------
    >>> from fafbseg import flywire
    >>> # Fetch supervoxel at two locations
    >>> locs = [[133131, 55615, 3289], [132802, 55661, 3289]]
    >>> flywire.locs_to_supervoxels(locs)
    array([79801454835332154, 79731086091150780], dtype=uint64)

    """
    if isinstance(locs, pd.DataFrame):
        if np.all(np.isin(['fw.x', 'fw.y', 'fw.z'], locs.columns)):
            locs = locs[['fw.x', 'fw.y', 'fw.z']].values
        elif np.all(np.isin(['x', 'y', 'z'], locs.columns)):
            locs = locs[['x', 'y', 'z']].values
        else:
            raise ValueError('`locs` as pandas.DataFrame must have either [fw.x'
                             ', fw.y, fw.z] or [x, y, z] columns.')

        # Make sure we are working with numbers
        if not np.issubdtype(locs.dtype, np.number):
            locs = locs.astype(np.float64)

    return spine.transform.get_segids(locs, segmentation='flywire_190410',
                                      coordinates=coordinates, mip=-1)


def neuron_to_segments(x, short=False, dataset='production', coordinates='voxel'):
    """Get root IDs overlapping with a given neuron.

    Parameters
    ----------
    x :                 Neuron/List
                        Neurons for which to return root IDs. Neurons must be
                        in FlyWire (FAFB14.1) space.
    short :             bool
                        If True will only return the top hit for each neuron
                        (including a confidence score).
    dataset :           str | CloudVolume
                        Against which FlyWire dataset to query::
                            - "production" (current production dataset, fly_v31)
                            - "sandbox" (i.e. fly_v26)

    coordinates :       "voxel" | "nm"
                        Units the neuron(s) are in. "voxel" is assumed to be
                        4x4x40 (x/y/z) nanometers.

    Returns
    -------
    overlap_matrix :    pandas.DataFrame
                        DataFrame of root IDs (rows) and IDs
                        (columns) with overlap in nodes as values::

                                 id     id1   id2
                            root_id
                            10336680915   5     0
                            10336682132   0     1

    summary :           pandas.DataFrame
                        If ``short=True``: DataFrame of top hits only::

                            id            match   confidence
                            12345   103366809155     0.87665
                            412314  103366821325     0.65233

    """
    if isinstance(x, navis.TreeNeuron):
        x = navis.NeuronList(x)

    assert isinstance(x, navis.NeuronList)

    # We must not perform this on x.nodes as this is a temporary property
    nodes = x.nodes

    # Get segmentation IDs
    nodes['root_id'] = locs_to_segments(nodes[['x', 'y', 'z']].values,
                                        coordinates=coordinates,
                                        root_ids=True, dataset=dataset)

    # Count segment IDs
    seg_counts = nodes.groupby(['neuron', 'root_id'], as_index=False).node_id.count()
    seg_counts.columns = ['id', 'root_id', 'counts']

    # Remove seg IDs 0
    seg_counts = seg_counts[seg_counts.root_id != 0]

    # Turn into matrix where columns are skeleton IDs, segment IDs are rows
    # and values are the overlap counts
    matrix = seg_counts.pivot(index='root_id', columns='id', values='counts')

    if not short:
        return matrix

    # Extract top IDs and scores
    top_id = matrix.index[np.argmax(matrix.fillna(0).values, axis=0)]

    # Confidence is the difference between top and 2nd score
    top_score = matrix.max(axis=0).values
    sec_score = np.sort(matrix.fillna(0).values, axis=0)[-2, :]
    conf = (top_score - sec_score) / matrix.sum(axis=0).values

    summary = pd.DataFrame([])
    summary['id'] = matrix.columns
    summary['match'] = top_id
    summary['confidence'] = conf

    return summary


def locs_to_segments(locs, root_ids=True, timestamp=None, dataset='production',
                     coordinates='voxel'):
    """Retrieve FlyWire segment IDs (root IDs) at given location(s).

    Parameters
    ----------
    locs :          list-like | pandas.DataFrame
                    Array of x/y/z coordinates. If DataFrame must contain
                    'x', 'y', 'z' or 'fw.x', 'fw.y', 'fw.z' columns. If both
                    present, 'fw.' columns take precedence)!
    root_ids :      bool
                    If True, will return root IDs. If False, will return supervoxel
                    IDs.
    timestamp :     int | str | datetime, optional
                    Get roots at given date (and time). Int must be unix
                    timestamp. String must be ISO 8601 - e.g. '2021-11-15'.
                    If timestamp is given, will ignore `use_cache`.
    dataset :       str | CloudVolume
                    Against which FlyWire dataset to query::
                        - "production" (current production dataset, fly_v31)
                        - "sandbox" (i.e. fly_v26)
                    Only relevant if ``root_ids=True``.
    coordinates :   "voxel" | "nm"
                    Units in which your coordinates are in. "voxel" is assumed
                    to be 4x4x40 (x/y/z) nanometers.

    Returns
    -------
    numpy.array
                List of segmentation IDs in the same order as ``locs``.

    Examples
    --------
    >>> from fafbseg import flywire
    >>> # Fetch root IDs at two locations
    >>> locs = [[133131, 55615, 3289], [132802, 55661, 3289]]
    >>> flywire.locs_to_segments(locs)
    array([720575940621039145, 720575940621039145])

    """
    svoxels = locs_to_supervoxels(locs, coordinates=coordinates)

    if not root_ids:
        return svoxels

    return supervoxels_to_roots(svoxels, timestamp=timestamp, dataset=dataset)


def skid_to_id(x,
               sample=None,
               dataset='production',
               progress=True, **kwargs):
    """Find the FlyWire root ID(s) corresponding to given CATMAID skeleton ID(s).

    This function works by:
        1. Fetch supervoxels for all nodes in the CATMAID skeletons
        2. Pick a random sample of ``sample`` of these supervoxels
        3. Fetch the most recent root IDs for the sample supervoxels
        4. Return the root ID that collectively cover 90% of the supervoxels

    Parameters
    ----------
    x :             int | list-like | str | TreeNeuron/List
                    Anything that's not a TreeNeuron/List will be passed
                    directly to ``pymaid.get_neuron``.
    sample :        int | float, optional
                    Number (>= 1) or fraction (< 1) of super nodes to sample
                    to find FlyWire IDs. If ``None`` (default), will use all
                    nodes.
    dataset :       str | CloudVolume
                    Against which FlyWire dataset to query::
                        - "production" (current production dataset, fly_v31)
                        - "sandbox" (i.e. fly_v26)
    progress :      bool
                    If True, shows progress bar.

    Returns
    -------
    pandas.DataFrame
                    Mapping of FlyWire root IDs to skeleton IDs with confidence::

                      skeleton_id   flywire_id   confidence
                    0
                    1

    """
    if not isinstance(x, (navis.TreeNeuron, navis.NeuronList)):
        x = pymaid.get_neuron(x)

    if isinstance(x, navis.NeuronList) and len(x) == 1:
        x = x[0]

    if isinstance(x, navis.NeuronList):
        res = []
        for n in navis.config.tqdm(x, desc='Searching',
                                   disable=not progress,
                                   leave=False):
            res.append(skid_to_id(n, dataset=dataset))
        return pd.concat(res, axis=0)
    elif isinstance(x, navis.TreeNeuron):
        nodes = x.nodes[['x', 'y', 'z']]
        if sample:
            if sample < 1:
                nodes = nodes.sample(frac=sample, random_state=1985)
            else:
                nodes = nodes.sample(n=sample, random_state=1985)
    else:
        raise TypeError(f'Unable to use data of type "{type(x)}"')

    # XForm coordinates from FAFB14 to FAFB14.1
    xformed = xform.fafb14_to_flywire(nodes[['x', 'y', 'z']].values,
                                      coordinates='nm')

    # Get the root IDs for each of these locations
    roots = locs_to_segments(xformed, coordinates='nm', dataset=dataset)

    # Drop zeros
    roots = roots[roots != 0]

    # Find unique Ids and count them
    unique, counts = np.unique(roots, return_counts=True)

    # Get sorted indices
    sort_ix = np.argsort(counts)

    # The "correct" ID is assumed to be the most frequent ID
    new_id = unique[sort_ix[-1]]

    # Confidence is the difference between the top and the 2nd most frequent ID
    if len(unique) > 1:
        diff_1st_2nd = counts[sort_ix[-1]] - counts[sort_ix[-2]]
        conf = round(diff_1st_2nd / roots.shape[0], 2)
    else:
        conf = 1

    return pd.DataFrame([[x.id, new_id, conf]],
                        columns=['skeleton_id', 'flywire_id', 'confidence'])


def is_latest_root(id, dataset='production', **kwargs):
    """Check if root is the current one.

    Parameters
    ----------
    id :            int | list-like
                    Single ID or list of FlyWire (root) IDs.
    dataset :       str | CloudVolume
                    Against which FlyWire dataset to query:
                      - "production" (current production dataset, fly_v31)
                      - "sandbox" (i.e. fly_v26)

    Returns
    -------
    numpy array
                    Array of booleans

    See Also
    --------
    :func:`~fafbseg.flywire.update_ids`
                    If you want the new ID.

    Examples
    --------
    >>> from fafbseg import flywire
    >>> flywire.is_latest_root(720575940621039145)
    array([True])

    """
    dataset = FLYWIRE_DATASETS.get(dataset, dataset)

    id = navis.utils.make_iterable(id).astype(str)

    # The server doesn't like being asked for zeros
    not_zero = id != '0'

    # Check if all other IDs are valid
    is_valid_root(id[not_zero], raise_exc=True)

    is_latest = np.ones(len(id)).astype(bool)

    session = requests.Session()
    token = get_chunkedgraph_secret()
    session.headers['Authorization'] = f"Bearer {token}"

    url = f'https://prod.flywire-daf.com/segmentation/api/v1/table/{dataset}/is_latest_roots?int64_as_str=1'
    post = {'node_ids': id[not_zero].tolist()}
    r = session.post(url, json=post)

    r.raise_for_status()

    is_latest[not_zero] = np.array(r.json()['is_latest'])

    return is_latest


def update_ids(id,
               sample=0.1,
               supervoxels=None,
               dataset='production',
               progress=True, **kwargs):
    """Retrieve the most recent version of given FlyWire (root) ID(s).

    This function works by:
        1. Check if ID is outdated (see :func:`fafbseg.flywire.is_latest_root`)
        2. If supervoxel provided, use it to update ID. Else try 3.
        3. See if we can map outdated IDs to a single up-to-date root (works
           if neuron has only seen merges). Else try 4.
        4. For uncertain IDs, fetch all supervoxels and pick a random sample
           (see ``sample`` parameter). Fetching the most recent root IDs for
           the sample of supervoxels and return the root ID that was hit the
           most often.

    Parameters
    ----------
    id :            int | list-like | DataFrame
                    Single ID or list of FlyWire (root) IDs. If DataFrame must
                    contain either a `root_id` or `root` column and optionally
                    a `supervoxel_id` or `supervoxel` column.
    sample :        int | float
                    Number (>= 1) or fraction (< 1) of super voxels to sample
                    to guess the most recent version.
    supervoxels :   int | list-like, optional
                    If provided will use these supervoxels to update ``id``
                    instead of sampling across all supervoxels.
    dataset :       str | CloudVolume
                    Against which FlyWire dataset to query:
                      - "production" (current production dataset, fly_v31)
                      - "sandbox" (i.e. fly_v26)
    progress :      bool
                    If True, shows progress bar.

    Returns
    -------
    pandas.DataFrame
                    Mapping of old -> new root IDs with confidence::

                      old_id   new_id   confidence   changed
                    0
                    1

    See Also
    --------
    :func:`~fafbseg.flywire.is_latest_root`
                    If all you want is to know whether a (root) ID is up-to-date.

    Examples
    --------
    >>> from fafbseg import flywire
    >>> flywire.update_ids(720575940621039145)
                   old_id              new_id  confidence  changed
    0  720575940621039145  720575940621039145           1    False

    """
    assert sample > 0, '`sample` must be > 0'

    # See if we already check if this was the latest root
    is_latest = kwargs.pop('is_latest', None)

    vol = parse_volume(dataset, **kwargs)

    if isinstance(id, pd.DataFrame):
        if isinstance(supervoxels, type(None)):
            if 'supervoxel_id' in id.columns:
                supervoxels = id['supervoxel_id'].values
            elif 'supervoxel' in id.columns:
                supervoxels = id['supervoxel'].values

        if 'root_id' in id.columns:
            id = id['root_id'].values
        elif 'root' in id.columns:
            id = id['root'].values
        else:
            raise ValueError('DataFrame must contain either `root_id` or '
                             '`root` column.')

    if isinstance(id, (list, set, np.ndarray)):
        # Run is_latest once for all roots
        is_latest = is_latest_root(id, dataset=dataset)
        if isinstance(supervoxels, type(None)):
            res = [update_ids(x,
                              dataset=dataset,
                              is_latest=il,
                              supervoxels=None,
                              sample=sample) for x, il, in navis.config.tqdm(zip(id, is_latest),
                                                                               desc='Updating',
                                                                               leave=False,
                                                                               total=len(id),
                                                                               disable=not progress or len(id) == 1)]
            res = pd.concat(res, axis=0, sort=False, ignore_index=True)
        else:
            if (not isinstance(supervoxels, (list, set, np.ndarray))
                or len(supervoxels) != len(id)):
                raise ValueError(f'Number of supervoxels ({len(supervoxels)}) does '
                                 f'not match number of root IDs ({len(id)})')
            elif any(pd.isnull(supervoxels)):
                raise ValueError('`supervoxels` must not contain `None`')
            elif any(pd.isnull(id)):
                raise ValueError('`id` must not contain `None`')

            id = np.array(id).astype(int)

            res = pd.DataFrame()
            res['old_id'] = id
            res['new_id'] = id
            res.loc[~is_latest, 'new_id'] = supervoxels_to_roots(supervoxels[~is_latest],
                                                                 dataset=dataset)
            res['conf'] = 1
            res['changed'] = res['new_id'] != res['old_id']
        return res

    try:
        id = int(id)
    except ValueError:
        raise ValueError(f'"{id} does not look like a valid root ID.')

    if id == 0 or pd.isnull(id):
        navis.config.logger.warning(f'Unable to update ID "{id}" - returning '
                                    'unchanged.')
        return id

    # Check if outdated
    if isinstance(is_latest, type(None)):
        is_latest = is_latest_root(id, dataset=dataset)[0]

    if not is_latest:
        client = get_cave_client(dataset=dataset)
        pot_roots = client.chunkedgraph.get_latest_roots(id)

        if len(pot_roots) == 1:
            new_id = pot_roots[0]
            conf = 1
        elif supervoxels:
            try:
                supervoxels = int(supervoxels)
            except ValueError:
                raise ValueError(f'"{supervoxels} does not look like a valid '
                                 'supervoxel ID.')
            new_id = client.chunkedgraph.get_root_id(supervoxels_to_roots)
            conf = 1
        else:
            # Get supervoxel ids - we need to use mip=0 because otherwise small
            # neurons might not have any (visible) supervoxels
            svoxels = roots_to_supervoxels(id, progress=False)[int(id)]

            # Note: instead of supervoxels we could also use higher level IDs
            # (stop layer 3 or 4) which might be much faster

            # Shuffle voxels
            np.random.shuffle(svoxels)

            # Generate sample
            if sample >= 1:
                smpl = svoxels[: sample]
            else:
                smpl = svoxels[: int(len(svoxels) * sample)]

            # Fetch up-to-date root IDs for the sampled supervoxels
            roots = supervoxels_to_roots(smpl, dataset=vol, progress=False)

            # Find unique Ids and count them
            unique, counts = np.unique(roots, return_counts=True)

            # Get sorted indices
            sort_ix = np.argsort(counts)

            # New Id is the most frequent ID
            new_id = unique[sort_ix[-1]]

            # Confidence is the difference between the top and the 2nd most
            # frequent ID
            if len(unique) > 1:
                conf = round((counts[sort_ix[-1]] - counts[sort_ix[-2]]) / sum(counts),
                             2)
            else:
                conf = 1
    else:
        new_id = id
        conf = 1

    return pd.DataFrame([[id, new_id, conf, id != new_id]],
                        columns=['old_id', 'new_id', 'confidence', 'changed']
                        ).astype({'old_id': int, 'new_id': int})


def snap_to_id(locs, id, snap_zero=False, dataset='production',
               search_radius=160, coordinates='nm', max_workers=4,
               verbose=True):
    """Snap locations to the correct segmentation ID.

    Works by:
     1. Fetch segmentation ID for each location and for those with the wrong ID:
     2. Fetch cube around each loc and snap to the closest voxel with correct ID

    Parameters
    ----------
    locs :          (N, 3) array
                    Array of x/y/z coordinates.
    id :            int
                    Expected ID at each location.
    snap_zero :     bool
                    If False (default), we will not snap locations that map to
                    segment ID 0 (i.e. no segmentation).
    dataset :       str | CloudVolume
                    Against which FlyWire dataset to query::
                        - "production" (current production dataset, fly_v31)
                        - "sandbox" (i.e. fly_v26)
    search_radius : int
                    Radius [nm] around a location to search for a position with
                    the correct ID. Lower values will be faster.
    coordinates :   "voxel" | "nm"
                    Coordinate system of `locs`. If "voxel" it is assumed to be
                    4 x 4 x 40 nm.
    max_workers :   int
    verbose :       bool
                    If True will plot summary at then end.

    Returns
    -------
    (N, 3) array
                x/y/z locations that are guaranteed to map to the correct ID.

    """
    assert coordinates in ['nm', 'nanometer', 'nanometers', 'voxel', 'voxels']

    if isinstance(locs, navis.TreeNeuron):
        locs = locs.nodes[['x', 'y', 'z']].values

    # This also makes sure we work on a copy
    locs = np.array(locs, copy=True)
    assert locs.ndim == 2 and locs.shape[1] == 3

    # From hereon out we are working with nanometers
    if coordinates in ('voxel', 'voxels'):
        locs *= [4, 4, 40]

    root_ids = locs_to_segments(locs, dataset=dataset, coordinates='nm')

    id_wrong = root_ids != id
    not_zero = root_ids != 0

    to_fix = id_wrong

    if not snap_zero:
        to_fix = to_fix & not_zero

    # Use parallel processes to go over the to-fix nodes
    with navis.config.tqdm(desc='Snapping', total=to_fix.sum(), leave=False) as pbar:
        with futures.ProcessPoolExecutor(max_workers=max_workers) as ex:
            loc_futures = [ex.submit(_process_cutout,
                                     id=id,
                                     loc=locs[ix],
                                     dataset=dataset,
                                     radius=search_radius) for ix in np.where(to_fix)[0]]
            for f in futures.as_completed(loc_futures):
                pbar.update(1)

    # Get results
    results = [f.result() for f in loc_futures]

    # Stack locations
    new_locs = np.vstack(results)

    # If no new location found, array will be [0, 0, 0]
    not_snapped = new_locs.max(axis=1) == 0

    # Update location
    to_update = np.where(to_fix)[0][~not_snapped]
    locs[to_update, :] = new_locs[~not_snapped]

    if verbose:
        msg = f"""\
        {to_fix.sum()} of {to_fix.shape[0]} locations needed to be snapped.
        Of these {not_snapped.sum()} locations could not be snapped - consider
        increasing `search_radius`.
        """
        print(textwrap.dedent(msg))

    return locs


def _process_cutout(loc, id, radius=160, dataset='production'):
    """Process single cutout for snap_to_id."""
    # Get this location
    loc = loc.round()

    # Generating bounding box around this location
    mn = loc - radius
    mx = loc + radius
    # Make sure it's a multiple of 4 and 40
    mn = mn - mn % [4, 4, 40]
    mx = mx - mx % [4, 4, 40]

    # Generate bounding box
    bbox = np.vstack((mn, mx))

    # Get the cutout, the resolution and offset
    cutout, res, offset_nm = get_segmentation_cutout(bbox,
                                                     dataset=dataset,
                                                     root_ids=True,
                                                     coordinates='nm')

    # Generate a mask
    mask = (cutout == id).astype(int, copy=False)

    # Erode so we move our point slightly more inside the segmentation
    mask = ndimage.binary_erosion(mask).astype(mask.dtype)

    # Find positions the ID we are looking for
    our_id = np.vstack(np.where(mask)).T

    # Return [0, 0, 0] if unable to snap (i.e. if id not within radius)
    if not our_id.size:
        return np.array([0, 0, 0])

    # Get the closest on to the center of the cutout
    center = np.divide(cutout.shape, 2).round()
    dist = np.abs(our_id - center).sum(axis=1)
    closest = our_id[np.argmin(dist)]

    # Convert the cutout offset to absolute 4/4/40 voxel coordinates
    snapped = closest * res + offset_nm

    return snapped


def get_segmentation_cutout(bbox, dataset='production', root_ids=True,
                            coordinates='voxel'):
    """Fetch cutout of segmentation.

    Parameters
    ----------
    bbox :          array-like
                    Bounding box for the cutout::

                        [[xmin, xmax], [ymin, ymax], [zmin, zmax]]

    root_ids :      bool
                    If True, will return root IDs. If False, will return
                    supervoxel IDs.
    dataset :       str | CloudVolume
                    Against which FlyWire dataset to query::
                        - "production" (current production dataset, fly_v31)
                        - "sandbox" (i.e. fly_v26)
    coordinates :   "voxel" | "nm"
                    Units in which your coordinates are in. "voxel" is assumed
                    to be 4x4x40 (x/y/z) nanometers.

    Returns
    -------
    cutout :        np.ndarry
                    (N, M) array of segmentation (root or supervoxel) IDs.
    resolution :    (3, ) numpy array
                    [x, y, z] resolution of voxel in cutout.
    nm_offset :     (3, ) numpy array
                    [x, y, z] offset in nanometers of the cutout with respect
                    to the absolute coordinates.

    """
    assert coordinates in ['nm', 'nanometer', 'nanometers', 'voxel', 'voxels']

    bbox = np.asarray(bbox)
    assert bbox.ndim == 2

    if bbox.shape == (2, 3):
        pass
    elif bbox.shape == (3, 2):
        bbox = bbox.T
    else:
        raise ValueError(f'`bbox` must have shape (2, 3) or (3, 2), got {bbox.shape}')

    vol = parse_volume(dataset)

    # First convert to nanometers
    if coordinates in ('voxel', 'voxels'):
        bbox = bbox * [4, 4, 40]

    # Now convert (back to) to [16, 16, 40] voxel
    bbox = (bbox / vol.scale['resolution']).round().astype(int)

    offset_nm = bbox[0] * vol.scale['resolution']

    # Get cutout
    cutout = vol[bbox[0][0]:bbox[1][0],
                 bbox[0][1]:bbox[1][1],
                 bbox[0][2]:bbox[1][2]]

    if root_ids:
        svoxels = np.unique(cutout.flatten())
        roots = supervoxels_to_roots(svoxels, dataset=vol)

        sv2r = dict(zip(svoxels[svoxels != 0], roots[svoxels != 0]))

        for k, v in sv2r.items():
            cutout[cutout == k] = v

    return cutout[:, :, :, 0], np.asarray(vol.scale['resolution']), offset_nm


def is_valid_root(x, raise_exc=False, dataset='production'):
    """Check if ID is (potentially) valid root ID.

    Parameters
    ----------
    x :             int | str | iterable
                    ID(s) to check.
    raise_exc :     bool
                    If True and any IDs are invalid will raise an error.
                    Mostly for internal use.

    Returns
    -------
    bool
            If ``x`` is a single ID.
    array
            If ``x`` is iterable.

    """
    vol = parse_volume(dataset)

    if navis.utils.is_iterable(x):
        is_valid =  np.array([is_valid_root(r, dataset=vol) for r in x])
        if raise_exc and not all(is_valid):
            invalid = set(np.asarray(x)[~is_valid].tolist())
            raise ValueError(f'Invalid root IDs found: {invalid}')
        return is_valid

    try:
        is_valid = vol.get_chunk_layer(x) == 10
    except:
        is_valid = False

    if raise_exc and not is_valid:
        raise ValueError(f'{x} is not a valid root ID')

    return is_valid


def is_valid_supervoxel(x, raise_exc=False, dataset='production'):
    """Check if ID is (potentially) valid supervoxel ID.

    Parameters
    ----------
    x :             int | str | iterable
                    ID(s) to check.
    raise_exc :     bool
                    If True and any IDs are invalid will raise an error.
                    Mostly for internal use.

    Returns
    -------
    bool
            If ``x`` is a single ID.
    array
            If ``x`` is iterable.

    """
    vol = parse_volume(dataset)

    if navis.utils.is_iterable(x):
        is_valid =  np.array([is_valid_supervoxel(r, dataset=vol) for r in x])
        if raise_exc and not all(is_valid):
            invalid = set(np.asarray(x)[~is_valid].tolist())
            raise ValueError(f'Invalid supervoxel IDs found: {invalid}')
        return is_valid

    try:
        is_valid = vol.get_chunk_layer(x) == 1
    except:
        is_valid = False

    if raise_exc and not is_valid:
        raise ValueError(f'{x} is not a valid supervoxel ID')

    return is_valid
