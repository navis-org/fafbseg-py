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

import datetime as dt
import cloudvolume as cv
import numpy as np
import pandas as pd

from concurrent import futures
from requests_futures.sessions import FuturesSession
from scipy import ndimage
from tqdm.auto import tqdm

from .. import utils
from .. import xform

from .utils import parse_volume, FLYWIRE_DATASETS

try:
    import skeletor as sk
except ImportError:
    sk = None
except BaseException:
    raise

__all__ = ['fetch_edit_history', 'fetch_leaderboard', 'locs_to_segments',
           'locs_to_supervoxels', 'skid_to_id', 'update_ids']


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
    >>> # Fetch leaderboard
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
    resp = [f.result() for f in tqdm(futures,
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
        df = df.fillna(0).astype(int)

        if df.shape[1] > 1:
            df.iloc[:, -1] -= df.iloc[:, :-1].sum(axis=1)

    df = df.iloc[:, ::-1]
    return df.loc[df.sum(axis=1).sort_values(ascending=False).index]


def fetch_edit_history(x, dataset='production', progress=True, max_threads=4):
    """Fetch edit history for given neuron(s).

    Parameters
    ----------
    x :             int | iterable
    dataset :       str | CloudVolume
                    Against which flywire dataset to query::
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
    token = cv.secrets.chunkedgraph_credentials['token']
    session.headers['Authorization'] = f"Bearer {token}"

    futures = []
    for id in x:
        dataset = FLYWIRE_DATASETS.get(dataset, dataset)
        url = f'https://prodv1.flywire-daf.com/segmentation/api/v1/table/{dataset}/root/{id}/tabular_change_log'
        f = future_session.get(url, params=None)
        futures.append(f)

    # Get the responses
    resp = [f.result() for f in tqdm(futures,
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


def locs_to_supervoxels(locs, mip=2, coordinates='pixel'):
    """Retrieve flywire supervoxel IDs at given location(s).

    Use Eric Perlman's service on spine.

    Parameters
    ----------
    locs :          list-like | pandas.DataFrame
                    Array of x/y/z coordinates. If DataFrame must contain
                    'x', 'y', 'z' or 'fw.x', 'fw.y', 'fw.z' columns. If both
                    present, 'fw.' columns take precedence)!
    mip :           int [2-8]
                    Scale to query. Lower mip = more precise but slower;
                    higher mip = faster but less precise (small supervoxels
                    might not show at all).
    coordinates :   "pixel" | "nm"
                    Units in which your coordinates are in. "pixel" is assumed
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

    return utils.query_spine(locs,
                             dataset='flywire_190410',
                             query='query',
                             coordinates=coordinates,
                             mip=2)


def locs_to_segments(locs, root_ids=True, dataset='production',
                     coordinates='pixel'):
    """Retrieve flywire segment IDs at given location(s).

    Parameters
    ----------
    locs :          list-like | pandas.DataFrame
                    Array of x/y/z coordinates. If DataFrame must contain
                    'x', 'y', 'z' or 'fw.x', 'fw.y', 'fw.z' columns. If both
                    present, 'fw.' columns take precedence)!
    root_ids :      bool
                    If True, will return root IDs. If False, will return supervoxel
                    IDs.
    dataset :       str | CloudVolume
                    Against which flywire dataset to query::
                        - "production" (current production dataset, fly_v31)
                        - "sandbox" (i.e. fly_v26)
                    Only relevant if ``root_ids=True``.
    coordinates :   "pixel" | "nm"
                    Units in which your coordinates are in. "pixel" is assumed
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

    vol = parse_volume(dataset)

    # get_roots() doesn't like to be asked for zeros - causes server error
    roots = np.zeros(svoxels.shape, dtype=np.int64)
    roots[svoxels != 0] = vol.get_roots(svoxels[svoxels != 0])

    return roots


def skid_to_id(x,
               dataset='production',
               progress=True, **kwargs):
    """Find the flywire ID(s) corresponding to given CATMAID skeleton ID(s).

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
    dataset :       str | CloudVolume
                    Against which flywire dataset to query::
                        - "production" (current production dataset, fly_v31)
                        - "sandbox" (i.e. fly_v26)
    progress :      bool
                    If True, shows progress bar.

    Returns
    -------
    pandas.DataFrame
                    Mapping of flywire IDs to skeleton IDs with confidence::

                      flywire_id   skeleton_id   confidence
                    0
                    1

    """
    vol = parse_volume(dataset, **kwargs)

    if not isinstance(x, (navis.TreeNeuron, navis.NeuronList)):
        x = pymaid.get_neuron(x)

    if isinstance(x, navis.TreeNeuron):
        nodes = x.nodes[['x', 'y', 'z']].copy()
        nodes['skeleton_id'] = x.id
    elif isinstance(x, navis.NeuronList):
        nodes = x.nodes[['x', 'y', 'z']].copy()
    else:
        raise TypeError(f'Unable to data of type "{type(x)}"')

    # XForm coordinates from FAFB14 to FAFB14.1
    xformed = xform.fafb14_to_flywire(nodes[['x', 'y', 'z']].values,
                                      coordinates='nm')

    # Get the root IDs for each of these locations
    roots = locs_to_segments(xformed, coordinates='nm')

    # Drop zeros
    roots = roots[roots != 0]

    # Find unique Ids and count them
    unique, counts = np.unique(roots, return_counts=True)

    # Get sorted indices
    sort_ix = np.argsort(counts)

    # New Id is the most frequent ID
    new_id = unique[sort_ix[-1]]

    # Confidence is the difference between the top and the 2nd most frequent ID
    if len(unique) > 1:
        diff_1st_2nd = counts[sort_ix[-1]] - counts[sort_ix[-2]]
        conf = round(diff_1st_2nd / roots.shape[0], 2)
    else:
        conf = 1

    return pd.DataFrame([[x.id, new_id, conf, x.id != new_id]],
                        columns=['old_id', 'new_id', 'confidence', 'changed'])


def update_ids(id,
               sample=0.1,
               dataset='production',
               progress=True, **kwargs):
    """Retrieve the most recent version of given flywire neuron(s).

    This function works by:
        1. Fetching all supervoxels for a given to-be-updated ID
        2. Picking a random sample of ``sample`` of these supervoxels
        3. Fetching the most recent root IDs for the sample supervoxels
        4. Returning the root ID that was hit the most.

    Parameters
    ----------
    id :            int | list-like
                    Single ID or list of flywire (root) IDs.
    sample :        int | float
                    Number (>= 1) or fraction (< 1) of super voxels to sample
                    to guess the most recent version.
    dataset :       str | CloudVolume
                    Against which flywire dataset to query:
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

    Examples
    --------
    >>> from fafbseg import flywire
    >>> flywire.update_ids(720575940621039145)
                   old_id              new_id  confidence  changed
    0  720575940621039145  720575940621039145           1    False

    """
    assert sample > 0, '`sample` must be > 0'

    vol = parse_volume(dataset, **kwargs)

    if isinstance(id, (list, set, np.ndarray)):
        res = [update_ids(x,
                          vol=vol,
                          sample=sample) for x in tqdm(id,
                                                       desc='Updating',
                                                       leave=False,
                                                       disable=not progress or len(id) == 1)]
        return pd.concat(res, axis=0, sort=False)

    # Get supervoxel ids - we need to use mip=0 because otherwise small neurons
    # might not have any (visible) supervoxels
    svoxels = vol.get_leaves(id, bbox=vol.meta.bounds(0), mip=0)

    # Shuffle voxels
    np.random.shuffle(svoxels)

    # Generate sample
    if sample >= 1:
        smpl = svoxels[: sample]
    else:
        smpl = svoxels[: int(len(svoxels) * sample)]

    # Fetch up-to-date root IDs for the sampled supervoxels
    roots = vol.get_roots(smpl)

    # Find unique Ids and count them
    unique, counts = np.unique(roots, return_counts=True)

    # Get sorted indices
    sort_ix = np.argsort(counts)

    # New Id is the most frequent ID
    new_id = unique[sort_ix[-1]]

    # Confidence is the difference between the top and the 2nd most frequent ID
    if len(unique) > 1:
        conf = round((counts[sort_ix[-1]] - counts[sort_ix[-2]]) / sum(counts),
                     2)
    else:
        conf = 1

    return pd.DataFrame([[id, new_id, conf, id != new_id]],
                        columns=['old_id', 'new_id', 'confidence', 'changed'])


def snap_to_id(locs, id, snap_zero=False, dataset='production',
               search_radius=160, coordinates='nm', max_workers=4,
               verbose=True):
    """Snap locations to the correct segmentation ID.

    Works by:
     1. Fetch segmentation ID for each location and for those with the wrong ID:
     2. Fetch cube around each loc and snap to the closest pixel with correct ID

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
                    Against which flywire dataset to query::
                        - "production" (current production dataset, fly_v31)
                        - "sandbox" (i.e. fly_v26)
    search_radius : int
                    Radius [nm] around a location to search for a position with
                    the correct ID. Lower values will be faster.
    coordinates :   "pixel" | "nm"
                    Coordinate system of `locs`. If "pixel" it is assumed to be
                    4 x 4 x 40 nm.
    max_workers :   int
    verbose :       bool
                    If True will plot summary at then end.

    Returns
    -------
    (N, 3) array
                x/y/z locations that are guaranteed to map to the correct ID.

    """
    assert coordinates in ['nm', 'pixel']

    if isinstance(locs, navis.TreeNeuron):
        locs = locs.nodes[['x', 'y', 'z']].values

    # This also makes sure we work on a copy
    locs = np.array(locs, copy=True)
    assert locs.ndim == 2 and locs.shape[1] == 3

    # From hereon out we are working with nanometers
    if coordinates == 'pixel':
        locs *= [4, 4, 40]

    root_ids = locs_to_segments(locs, dataset=dataset, coordinates='nm')

    id_wrong = root_ids != id
    not_zero = root_ids != 0

    to_fix = id_wrong

    if not snap_zero:
        to_fix = to_fix & not_zero

    # Use parallel processes to go over the to-fix nodes
    with tqdm(desc='Snapping', total=to_fix.sum(), leave=False) as pbar:
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
    mask = (cutout == id).astype(int)

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

    # Convert the cutout offset to absolute 4/4/40 pixel coordinates
    snapped = closest * res + offset_nm

    return snapped


def get_segmentation_cutout(bbox, dataset='production', root_ids=True,
                            coordinates='pixel'):
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
                    Against which flywire dataset to query::
                        - "production" (current production dataset, fly_v31)
                        - "sandbox" (i.e. fly_v26)
    coordinates :   "pixel" | "nm"
                    Units in which your coordinates are in. "pixel" is assumed
                    to be 4x4x40 (x/y/z) nanometers.

    Returns
    -------
    cutout :        np.ndarry
                    (N, M) array of segmentation (root or supervoxel) IDs.
    resolution :    (3, ) numpy array
                    [x, y, z] resolution of pixels in cutout.
    nm_offset :     (3, ) numpy array
                    [x, y, z] offset in nanometers of the cutout with respect
                    to the absolute coordinates.

    """
    assert coordinates in ['nm', 'pixel']

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
    if coordinates == 'pixel':
        bbox = bbox * [4, 4, 40]

    # Now convert (back to) to [16, 16, 40] pixel
    bbox = (bbox / vol.scale['resolution']).round().astype(int)

    offset_nm = bbox[0] * vol.scale['resolution']

    # Get cutout
    cutout = vol[bbox[0][0]:bbox[1][0],
                 bbox[0][1]:bbox[1][1],
                 bbox[0][2]:bbox[1][2]]

    if root_ids:
        svoxels = np.unique(cutout.flatten())
        roots = vol.get_roots(svoxels[svoxels != 0])

        sv2r = dict(zip(svoxels[svoxels != 0], roots))

        for k, v in sv2r.items():
            cutout[cutout == k] = v

    return cutout[:, :, :, 0], np.asarray(vol.scale['resolution']), offset_nm
