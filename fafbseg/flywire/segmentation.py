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
import copy

import cloudvolume as cv
import datetime as dt
import numpy as np
import pandas as pd
import networkx as nx

from concurrent import futures
from diskcache import Cache
from requests_futures.sessions import FuturesSession
from scipy import ndimage
from tqdm.auto import tqdm

from .. import spine
from .. import xform

from ..utils import make_iterable, GSPointLoader
from .utils import (
    get_cloudvolume,
    FLYWIRE_DATASETS,
    get_chunkedgraph_secret,
    retry,
    get_cave_client,
    parse_bounds,
    package_timestamp,
    inject_dataset,
)
from .annotations import parse_neuroncriteria


__all__ = [
    "get_edit_history",
    "get_leaderboard",
    "locs_to_segments",
    "locs_to_supervoxels",
    "skid_to_id",
    "update_ids",
    "roots_to_supervoxels",
    "supervoxels_to_roots",
    "neuron_to_segments",
    "is_latest_root",
    "is_valid_root",
    "is_valid_supervoxel",
    "get_voxels",
    "get_lineage_graph",
    "find_common_time",
    "get_segmentation_cutout",
]


@inject_dataset()
def get_lineage_graph(
    x,
    size=False,
    user=False,
    synapses=False,
    proofreading_status=False,
    progress=True,
    *,
    dataset=None,
):
    """Get lineage graph for given neuron.

    This piggy-backs on the CAVEclient but importantly we remap users and
    operation IDs such that each node's labels refer to the operation that
    created them.

    Parameters
    ----------
    x :         int
                A single root ID.
    size :      bool
                If True, will add `size` and `survivals` node attributes. The
                former indicates the number of supervoxels, the latter how many
                of these supervoxels made it into `x`.
    synapses :  bool
                If True, will add `pre|post|synapses` node attributes which
                indicate how many of the synapses in `x` came from this fragment.
                Note that this doesn't tell you e.g. how many false-positive
                synapses were removed via a split. This works only if the root
                ID is up-to-date.
    user :      bool
                If True, will add user `user` node attribute.
    proofreading_status : bool
                If True, will add a `proofread_by` node attribute indicating if
                a user has set a given root ID to proofread.

    Returns
    -------
    networkx.DiGraph

    """
    x = np.int64(x)

    client = get_cave_client(dataset=dataset)
    G = client.chunkedgraph.get_lineage_graph(x, as_nx_graph=True)

    # Remap operation ID
    op_remapped = {}
    for n in G:
        pred = list(G.predecessors(n))
        if pred:
            op_remapped[n] = G.nodes[pred[0]]["operation_id"]

    # Remove existing operation IDs
    for n in G.nodes:
        G.nodes[n].pop("operation_id", None)
    # Apply new IDs
    nx.set_node_attributes(G, op_remapped, name="operation_id")

    if user:
        op_ids = nx.get_node_attributes(G, "operation_id")
        details = client.chunkedgraph.get_operation_details(list(op_ids.values()))
        users = {n: details[str(o)]["user"] for n, o in op_ids.items()}
        nx.set_node_attributes(G, users, name="user")

    if size:
        sv = roots_to_supervoxels(list(G.nodes), dataset=dataset, progress=progress)
        sizes = {n: len(sv[n]) for n in G.nodes}
        nx.set_node_attributes(G, sizes, name="size")

        survivors = {n: int(np.isin(sv[n], sv[x]).sum()) for n in G.nodes}
        nx.set_node_attributes(G, survivors, name="survivors")
    else:
        sv = None

    if synapses:
        pre = client.materialize.live_query(
            table=client.materialize.synapse_table,
            filter_equal_dict=dict(pre_pt_root_id=x),
            timestamp=dt.datetime.now(),
            select_columns=["pre_pt_supervoxel_id", "post_pt_supervoxel_id"],
        )
        post = client.materialize.live_query(
            table=client.materialize.synapse_table,
            filter_equal_dict=dict(post_pt_root_id=x),
            timestamp=dt.datetime.now(),
            select_columns=["pre_pt_supervoxel_id", "post_pt_supervoxel_id"],
        )
        if isinstance(sv, type(None)):
            sv = roots_to_supervoxels(list(G.nodes), dataset=dataset, progress=progress)

        n_pre = {n: int(pre.pre_pt_supervoxel_id.isin(sv[n]).sum()) for n in G.nodes}
        n_post = {n: int(post.post_pt_supervoxel_id.isin(sv[n]).sum()) for n in G.nodes}
        n_syn = {n: n_pre[n] + n_post[n] for n in G.nodes}
        nx.set_node_attributes(G, n_pre, name="presynapses")
        nx.set_node_attributes(G, n_post, name="postsynapses")
        nx.set_node_attributes(G, n_syn, name="synapses")

    if proofreading_status:
        from .annotations import get_cave_table

        nodes = np.array(list(G.nodes), dtype=np.int64)
        pr = get_cave_table(
            "proofreading_status_public_v1", filter_in_dict=dict(valid_id=nodes)
        )
        if len(pr):
            user = pr.groupby("valid_id").user_id.apply(list).to_dict()
            nx.set_node_attributes(
                G, {n: user[n] for n in pr.valid_id}, name="proofread_by"
            )

    return G


def get_leaderboard(days=7, by_day=False, progress=True, max_threads=4):
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
    >>> hist = flywire.get_leaderboard(by_day=True)              #doctest: +SKIP
    >>> # Plot user actions over time
    >>> hist.T.plot()                                            #doctest: +SKIP

    """
    assert isinstance(days, (int, np.integer))
    assert days >= 0

    session = requests.Session()
    if not by_day:
        url = f"https://pyrdev.eyewire.org/flywire-leaderboard?days={days-1}"
        resp = session.get(url, params=None)
        resp.raise_for_status()
        return pd.DataFrame.from_records(resp.json()["entries"]).set_index("name")

    future_session = FuturesSession(session=session, max_workers=max_threads)
    futures = []
    for i in range(0, days):
        url = f"https://pyrdev.eyewire.org/flywire-leaderboard?days={i}"
        futures.append(future_session.get(url, params=None))

    # Get the responses
    resp = [
        f.result()
        for f in navis.config.tqdm(
            futures,
            desc="Fetching",
            disable=not progress or len(futures) == 1,
            leave=False,
        )
    ]

    df = None
    for i, r in enumerate(resp):
        date = dt.date.today() - dt.timedelta(days=i)
        r.raise_for_status()
        this_df = pd.DataFrame.from_records(r.json()["entries"]).set_index("name")
        this_df.columns = [date]
        if isinstance(df, type(None)):
            df = this_df
        else:
            df = pd.merge(df, this_df, how="outer", left_index=True, right_index=True)

    # Make sure we don't have NAs
    df = df.fillna(0).astype(int)

    # This breaks it down into days
    if df.shape[1] > 1:
        df.iloc[:, 1:] = df.iloc[:, 1:].values - df.iloc[:, :-1].values

    # Reverse such that the right-most entry is the current date
    df = df.iloc[:, ::-1]
    return df.loc[df.sum(axis=1).sort_values(ascending=False).index]


@parse_neuroncriteria()
@inject_dataset()
def get_edit_history(x, progress=True, errors="raise", max_threads=4, *, dataset=None):
    """Fetch edit history for given neuron(s).

    Note that neurons that haven't seen any edits will simply not show up in
    returned table.

    Parameters
    ----------
    x :             int | list of int | NeuronCriteria
                    Segmentation (root) ID(s).
    progress :      bool
                    If True, show progress bar.
    max_threads :   int
                    Max number of parallel requests to server.
    dataset :       "public" | "production" | "sandbox" | "flat_630", optional
                    Against which FlyWire dataset to query. If ``None`` will fall
                    back to the default dataset (see
                    :func:`~fafbseg.flywire.set_default_dataset`).

    Returns
    -------
    pandas.DataFrame

    Examples
    --------
    >>> from fafbseg import flywire
    >>> # Fetch edits
    >>> edits = flywire.get_edit_history(720575940621039145)
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
    session.headers["Authorization"] = f"Bearer {token}"

    futures = []
    for id in x:
        dataset = FLYWIRE_DATASETS.get(dataset, dataset)
        url = f"https://prod.flywire-daf.com/segmentation/api/v1/table/{dataset}/root/{id}/tabular_change_log"
        f = future_session.get(url, params=None)
        futures.append(f)

    # Get the responses
    resp = [
        f.result()
        for f in navis.config.tqdm(
            futures,
            desc="Fetching",
            disable=not progress or len(futures) == 1,
            leave=False,
        )
    ]

    df = []
    for r, i in zip(resp, x):
        # Code 500 means server error
        if r.status_code == 500:
            # If server responds a time-out, it means that the root ID has not
            # seen any edits from base segmentation.
            if "Read timed out" in r.json().get("message", ""):
                continue

        try:
            r.raise_for_status()
        except BaseException:
            if errors == "raise":
                raise
            else:
                print(f"Error fetching logs for {i}")
                continue

        this_df = pd.DataFrame(r.json())
        this_df["segment"] = i
        df.append(this_df)

    # Concat if any edits at all
    if any([not f.empty for f in df]):
        # Drop neurons without edits
        df = [f for f in df if not f.empty]
        df = pd.concat(df, axis=0, sort=True)
        df["timestamp"] = pd.to_datetime(df.timestamp, unit="ms")
    else:
        # Return the first empty data frame
        df = df[0]

    return df


@parse_neuroncriteria()
@inject_dataset(disallowed=["flat_630", "flat_571"])
def roots_to_supervoxels(x, use_cache=True, progress=True, *, dataset=None):
    """Get supervoxels making up given neurons.

    Parameters
    ----------
    x :             int | list of int | NeuronCriteria
                    Segmentation (root) ID(s).
    use_cache :     bool
                    Whether to use disk cache to avoid repeated queries for the
                    same root. Cache is stored in `~/.fafbseg/`.
    progress :      bool
                    If True, show progress bar.
    dataset :       "public" | "production" | "sandbox", optional
                    Against which FlyWire dataset to query. If ``None`` will fall
                    back to the default dataset (see
                    :func:`~fafbseg.flywire.set_default_dataset`).

    Returns
    -------
    dict
                    ``{root_id: [supervoxel_id1, ssupervoxel_id2, ...], ...}``

    Examples
    --------
    >>> from fafbseg import flywire
    >>> flywire.roots_to_supervoxels(720575940619164912)[720575940619164912]
    array([78251074787604983, 78251074787607484, 78251074787605192, ...,
           78673699569883003, 78673699569870455, 78673699569887289],
           dtype=uint64)

    """
    # Make sure we are working with an array of integers
    x = make_iterable(x, force_type=np.int64)

    # Make sure we're not getting bogged down with duplicates
    x = np.unique(x)

    if len(x) <= 1:
        progress = False

    # Get the volume
    vol = get_cloudvolume(dataset)

    svoxels = {}
    # See what we can get from cache
    if use_cache:
        # Cache for root -> supervoxels
        # Grows to max 1Gb by default and persists across sessions
        with Cache(directory="~/.fafbseg/svoxel_cache/") as sv_cache:
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
    miss = x[~np.isin(x, np.array(list(svoxels.keys()), dtype=np.int64))]
    get_leaves = retry(vol.get_leaves)
    with navis.config.tqdm(
        desc="Querying", total=len(x), disable=not progress, leave=False
    ) as pbar:
        # Update for those for which we had cached data
        pbar.update(len(svoxels))

        for i in miss:
            svoxels[i] = get_leaves(i, bbox=vol.meta.bounds(0), mip=0)
            pbar.update()

    # Update cache
    if use_cache:
        with Cache(directory="~/.fafbseg/svoxel_cache/") as sv_cache:
            with sv_cache.transact():
                for i in miss:
                    sv_cache[i] = svoxels[i]

    return svoxels


@inject_dataset(disallowed=["flat_630", "flat_571"])
def supervoxels_to_roots(
    x,
    timestamp=None,
    batch_size=10_000,
    stop_layer=10,
    retry=True,
    progress=True,
    *,
    dataset=None,
):
    """Get root(s) for given supervoxel(s).

    Parameters
    ----------
    x :             int | list of int
                    Supervoxel ID(s) to find the root(s) for. Also works for
                    e.g. L2 IDs.
    timestamp :     int | str | datetime | "mat", optional
                    Get roots at given date (and time). Int must be unix
                    timestamp. String must be ISO 8601 - e.g. '2021-11-15'.
                    "mat" will use the timestamp of the most recent
                    materialization. You can also use e.g. "mat_438" to get the
                    root ID at a specific materialization.
    batch_size :    int
                    Max number of supervoxel IDs per query. Reduce batch size if
                    you experience time outs.
    stop_layer :    int
                    Set e.g. to ``2`` to get L2 IDs instead of root IDs.
    retry :         bool
                    Whether to retry if a batched query fails.
    progress :      bool
                    If True, show progress bar.
    dataset :       "public" | "production" | "sandbox", optional
                    Against which FlyWire dataset to query. If ``None`` will fall
                    back to the default dataset (see
                    :func:`~fafbseg.flywire.set_default_dataset`).

    Returns
    -------
    roots  :        numpy array
                    Roots corresponding to supervoxels in `x`.

    Examples
    --------
    >>> from fafbseg import flywire
    >>> flywire.supervoxels_to_roots(78321855915861142)
    array([720575940594028562])

    """
    # Make sure we are working with an array of integers
    x = make_iterable(x, force_type=np.int64)

    # Check if IDs are valid (zeros are fine because we filter for them later on)
    # is_valid_supervoxel(x[(x != 0) & (x != '0')], raise_exc=True)

    # Parse the volume
    vol = get_cloudvolume(dataset)

    # Prepare results array
    roots = np.zeros(x.shape, dtype=np.int64)

    if isinstance(timestamp, str) and timestamp.startswith("mat"):
        client = get_cave_client(dataset=dataset)
        if timestamp == "mat" or timestamp == "mat_latest":
            timestamp = client.materialize.get_timestamp()
        else:
            # Split e.g. 'mat_432' to extract version and query timestamp
            version = int(timestamp.split("_")[1])
            timestamp = client.materialize.get_timestamp(version)

    if isinstance(timestamp, np.datetime64):
        timestamp = str(timestamp)

    with tqdm(
        desc="Fetching roots",
        leave=False,
        total=len(x),
        disable=not progress or len(x) < batch_size,
    ) as pbar:
        for i in range(0, len(x), int(batch_size)):
            # This batch
            batch = x[i : i + batch_size]

            # get_roots() doesn't like to be asked for zeros - causes server error
            not_zero = batch != 0
            try:
                roots[i : i + batch_size][not_zero] = vol.get_roots(
                    batch[not_zero], stop_layer=stop_layer, timestamp=timestamp
                )
            except KeyboardInterrupt:
                raise
            except BaseException:
                if not retry:
                    raise
                time.sleep(1)
                roots[i : i + batch_size][not_zero] = vol.get_roots(
                    batch[not_zero], stop_layer=stop_layer, timestamp=timestamp
                )

            pbar.update(len(batch))

    return roots


def locs_to_supervoxels(locs, mip=2, coordinates="voxel", backend="spine"):
    """Retrieve FlyWire supervoxel IDs at given location(s).

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
    backend :       "spine" | "cloudvolume"
                    Which backend to use. Use "cloudvolume" only when spine
                    doesn't work.

    Returns
    -------
    numpy.array
                List of segmentation IDs in the same order as ``locs``. Invalid
                locations will be returned with ID 0.

    See Also
    --------
    :func:`~fafbseg.flywire.locs_to_segments`
                Takes locations and returns root IDs. Can also map to a specific
                time or materialization.

    Examples
    --------
    >>> from fafbseg import flywire
    >>> # Fetch supervoxel at two locations
    >>> locs = [[133131, 55615, 3289], [132802, 55661, 3289]]
    >>> flywire.locs_to_supervoxels(locs)
    array([79801454835332154, 79731086091150780], dtype=uint64)

    """
    if backend not in ("spine", "cloudvolume"):
        raise ValueError(f"`backend` not recognised: {backend}")

    if isinstance(locs, pd.DataFrame):
        if np.all(np.isin(["fw.x", "fw.y", "fw.z"], locs.columns)):
            locs = locs[["fw.x", "fw.y", "fw.z"]].values
        elif np.all(np.isin(["x", "y", "z"], locs.columns)):
            locs = locs[["x", "y", "z"]].values
        else:
            raise ValueError(
                "`locs` as pandas.DataFrame must have either [fw.x"
                ", fw.y, fw.z] or [x, y, z] columns."
            )

        # Make sure we are working with numbers
        if not np.issubdtype(locs.dtype, np.number):
            locs = locs.astype(np.float64)

    if backend == "spine":
        return spine.transform.get_segids(
            locs, segmentation="flywire_190410", coordinates=coordinates, mip=mip
        )
    else:
        vol = copy.deepcopy(get_cloudvolume("production"))
        # Lower mips appear to cause inconsistencies despite spine also only
        # using mip 2 (IIRC?)
        # vol.mip = 2
        pl = GSPointLoader(vol)

        if coordinates in ("voxel", "voxels"):
            locs = locs * [4, 4, 40]

        pl.add_points(locs)

        points, data = pl.load_all(max_workers=4, progress=True, return_sorted=True)

        return data


@inject_dataset()
def neuron_to_segments(x, short=False, coordinates="voxel", *, dataset=None):
    """Get root IDs overlapping with a given neuron.

    Parameters
    ----------
    x :             Neuron/List
                    Neurons for which to return root IDs. Neurons must be
                    in FlyWire (FAFB14.1) space.
    short :         bool
                    If True will only return the top hit for each neuron
                    (including a confidence score).
    coordinates :   "voxel" | "nm"
                    Units the neuron(s) are in. "voxel" is assumed to be
                    4x4x40 (x/y/z) nanometers.
    dataset :       "public" | "production" | "sandbox" | "flat_630", optional
                    Against which FlyWire dataset to query. If ``None`` will fall
                    back to the default dataset (see
                    :func:`~fafbseg.flywire.set_default_dataset`).

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

    See Also
    --------
    :func:`~fafbseg.flywire.skid_to_id`
                    Takes a CATMAID (FAFB) skeleton ID or annotations and returns
                    corresponding FlyWire root IDs.

    """
    if isinstance(x, navis.TreeNeuron):
        x = navis.NeuronList(x)

    assert isinstance(x, navis.NeuronList)

    # We must not perform this on x.nodes as this is a temporary property
    nodes = x.nodes

    # Get segmentation IDs
    nodes["root_id"] = locs_to_segments(
        nodes[["x", "y", "z"]].values, coordinates=coordinates, dataset=dataset
    )

    # Count segment IDs
    seg_counts = nodes.groupby(["neuron", "root_id"], as_index=False).node_id.count()
    seg_counts.columns = ["id", "root_id", "counts"]

    # Remove seg IDs 0
    seg_counts = seg_counts[seg_counts.root_id != 0]

    # Turn into matrix where columns are skeleton IDs, segment IDs are rows
    # and values are the overlap counts
    matrix = seg_counts.pivot(index="root_id", columns="id", values="counts")

    if not short:
        return matrix

    # Extract top IDs and scores
    top_id = matrix.index[np.argmax(matrix.fillna(0).values, axis=0)]

    # Confidence is the difference between top and 2nd score
    top_score = matrix.max(axis=0).values
    sec_score = np.sort(matrix.fillna(0).values, axis=0)[-2, :]
    conf = (top_score - sec_score) / matrix.sum(axis=0).values

    summary = pd.DataFrame([])
    summary["id"] = matrix.columns
    summary["match"] = top_id
    summary["confidence"] = conf

    return summary


@inject_dataset(disallowed=["flat_630", "flat_571"])
def locs_to_segments(
    locs, timestamp=None, backend="spine", coordinates="voxel", *, dataset=None
):
    """Retrieve FlyWire segment (i.e. root) IDs at given location(s).

    Parameters
    ----------
    locs :          list-like | pandas.DataFrame
                    Array of x/y/z coordinates. If DataFrame must contain
                    'x', 'y', 'z' or 'fw.x', 'fw.y', 'fw.z' columns. If both
                    present, 'fw.' columns take precedence)!
    timestamp :     int | str | datetime | "mat", optional
                    Get roots at given date (and time). Int must be unix
                    timestamp. String must be ISO 8601 - e.g. '2021-11-15'.
                    "mat" will use the timestamp of the most recent
                    materalization. You can also use e.g. "mat_438" to get the
                    root ID at a specific materialization.
    backend :       "spine" | "cloudvolume"
                    Which backend to use. Use "cloudvolume" only when spine
                    doesn't work because it's terribly slow.
    coordinates :   "voxel" | "nm"
                    Units in which your coordinates are in. "voxel" is assumed
                    to be 4x4x40 (x/y/z) nanometers.
    dataset :       "public" | "production" | "sandbox" | "flat_630", optional
                    Against which FlyWire dataset to query. If ``None`` will fall
                    back to the default dataset (see
                    :func:`~fafbseg.flywire.set_default_dataset`). Only relevant
                    if ``root_ids=True``.

    Returns
    -------
    numpy.array
                    List of segmentation IDs in the same order as ``locs``.

    See Also
    --------
    :func:`~fafbseg.flywire.locs_to_supervoxels`
                    Takes locations and returns supervoxel IDs.

    Examples
    --------
    >>> from fafbseg import flywire
    >>> # Fetch root IDs at two locations
    >>> locs = [[133131, 55615, 3289], [132802, 55661, 3289]]
    >>> flywire.locs_to_segments(locs)
    array([720575940631693610, 720575940631693610])

    """
    svoxels = locs_to_supervoxels(locs, coordinates=coordinates, backend=backend)

    return supervoxels_to_roots(svoxels, timestamp=timestamp, dataset=dataset)


@inject_dataset()
def skid_to_id(x, sample=None, catmaid_instance=None, progress=True, *, dataset=None):
    """Find the FlyWire root ID for a given (FAFB) CATMAID neuron.

    This function works by:
        1. Fetch the skeleton for given CATMAID neuron.
        2. Transform the skeleton to FlyWire space.
        3. Map the x/y/z location of the skeleton nodes to root IDs.
        4. Return the root ID that was seen the most often.

    Parameters
    ----------
    x :             int | list-like | str | TreeNeuron/List
                    Anything that's not a TreeNeuron/List will be passed
                    directly to ``pymaid.get_neuron``.
    sample :        int | float, optional
                    Number (>= 1) or fraction (< 1) of skeleton nodes to sample
                    to find FlyWire root IDs. If ``None`` (default), will use
                    all nodes.
    catmaid_instance : pymaid.CatmaidInstance, optional
                    Connection to a CATMAID server. If ``None``, will use the
                    current global connection. See pymaid docs for details.
    progress :      bool
                    If True, shows progress bar.
    dataset :       "public" | "production" | "sandbox" | "flat_630", optional
                    Against which FlyWire dataset to query. If ``None`` will fall
                    back to the default dataset (see
                    :func:`~fafbseg.flywire.set_default_dataset`).

    Returns
    -------
    pandas.DataFrame
                    Mapping of skeleton IDs to FlyWire root IDs. Confidence
                    is the difference between the frequency of the root ID that
                    was seen most often and the second most seen ID.

    See Also
    --------
    :func:`~fafbseg.flywire.neuron_to_segments`
                    Takes already downloaded and transformed neuron(s) and
                    returns corresponding FlyWire root IDs.

    Examples
    --------
    >>> from fafbseg import flywire
    >>> import pymaid
    >>> # Connect to the VFB's CATMAID
    >>> rm = pymaid.CatmaidInstance('https://fafb.catmaid.virtualflybrain.org/',
    ...                             project_id=1, api_token=None)
    >>> roots = flywire.skid_to_id([6762, 2379517])
    >>> roots
      skeleton_id          flywire_id  confidence
    0        6762  720575940608544011        0.80
    1     2379517  720575940617229632        0.42

    """
    if not isinstance(x, (navis.TreeNeuron, navis.NeuronList)):
        x = pymaid.get_neuron(x, remote_instance=catmaid_instance)

    if isinstance(x, navis.NeuronList) and len(x) == 1:
        x = x[0]

    if isinstance(x, navis.NeuronList):
        res = []
        for n in navis.config.tqdm(
            x, desc="Searching", disable=not progress, leave=False
        ):
            res.append(skid_to_id(n, dataset=dataset))
        return pd.concat(res, axis=0).reset_index(drop=True)
    elif isinstance(x, navis.TreeNeuron):
        nodes = x.nodes[["x", "y", "z"]]
        if sample:
            if sample < 1:
                nodes = nodes.sample(frac=sample, random_state=1985)
            else:
                nodes = nodes.sample(n=sample, random_state=1985)
    else:
        raise TypeError(f'Unable to use data of type "{type(x)}"')

    # XForm coordinates from FAFB14 to FAFB14.1
    xformed = xform.fafb14_to_flywire(nodes[["x", "y", "z"]].values, coordinates="nm")

    # Get the root IDs for each of these locations
    roots = locs_to_segments(xformed, coordinates="nm", dataset=dataset)

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

    return pd.DataFrame(
        [[x.id, new_id, conf]], columns=["skeleton_id", "flywire_id", "confidence"]
    )


@inject_dataset(disallowed=["flat_630", "flat_571"])
@retry
def is_latest_root(id, timestamp=None, progress=True, *, dataset=None, **kwargs):
    """Check if root is the current one.

    Parameters
    ----------
    id :            int | list-like
                    Single ID or list of FlyWire (root) IDs.
    timestamp :     int | str | datetime | "mat", optional
                    Checks if roots existed at given date (and time). Int must
                    be unix timestamp. String must be ISO 8601 - e.g. '2021-11-15'.
                    "mat" will use the timestamp of the most recent
                    materialization. You can also use e.g. "mat_438" to get the
                    root ID at a specific materialization.
    progress :      bool
                    Whether to show progress bar.
    dataset :       "public" | "production" | "sandbox", optional
                    Against which FlyWire dataset to query. If ``None`` will fall
                    back to the default dataset (see
                    :func:`~fafbseg.flywire.set_default_dataset`).

    Returns
    -------
    numpy array
                    Array of booleans

    See Also
    --------
    :func:`~fafbseg.flywire.update_ids`
                    If you want the new ID. Also allows mapping to a specific
                    time or materialization.

    Examples
    --------
    >>> from fafbseg import flywire
    >>> flywire.is_latest_root(720575940631693610)
    array([ True])

    """
    id = make_iterable(id, force_type=str)

    # The server doesn't like being asked for zeros
    not_zero = id != "0"

    # Check if all other IDs are valid
    is_valid_root(id[not_zero], raise_exc=True, dataset=dataset)

    is_latest = np.ones(len(id)).astype(bool)

    client = get_cave_client(dataset=dataset)
    session = requests.Session()
    token = get_chunkedgraph_secret()
    session.headers["Authorization"] = f"Bearer {token}"
    url = (
        client.chunkedgraph._endpoints["is_latest_roots"].format_map(client.chunkedgraph.default_url_mapping)
    )

    if isinstance(timestamp, str) and timestamp.startswith("mat"):        
        if timestamp == "mat" or timestamp == "mat_latest":
            timestamp = client.materialize.get_timestamp()
        else:
            # Split e.g. 'mat_432' to extract version and query timestamp
            version = int(timestamp.split("_")[1])
            timestamp = client.materialize.get_timestamp(version)

    if isinstance(timestamp, np.datetime64):
        timestamp = str(timestamp)

    if isinstance(timestamp, str):
        timestamp = dt.datetime.fromisoformat(timestamp)

    if timestamp is not None:
        params = package_timestamp(timestamp)
    else:
        params = None

    batch_size = 100_000
    with navis.config.tqdm(
        desc="Checking",
        total=not_zero.sum(),
        disable=(not_zero.sum() <= batch_size) or not progress,
        leave=False,
    ) as pbar:
        for i in range(0, not_zero.sum(), batch_size):
            batch = id[not_zero][i : i + batch_size]
            post = {"node_ids": batch.tolist()}

            # Update progress bar
            pbar.update(len(batch))

            r = session.post(url, json=post, params=params)

            r.raise_for_status()

            is_latest[np.where(not_zero)[0][i : i + batch_size]] = np.array(
                r.json()["is_latest"]
            )

    return is_latest


@parse_neuroncriteria()
@inject_dataset(disallowed=["flat_630", "flat_571"])
def find_common_time(root_ids, progress=True, *, dataset=None):
    """Find a time at which given root IDs co-existed.

    Parameters
    ----------
    root_ids :      list | np.ndarray | NeuronCriteria
                    Root IDs to check.
    progress :      bool
                    If True, shows progress bar.
    dataset :       "public" | "production" | "sandbox", optional
                    Against which FlyWire dataset to query. If ``None`` will fall
                    back to the default dataset (see
                    :func:`~fafbseg.flywire.set_default_dataset`).

    Returns
    -------
    datetime.datetime

    """
    root_ids = np.asarray(root_ids, dtype=np.int64)

    client = get_cave_client(dataset=dataset)

    # Get timestamps when roots were created
    creations = client.chunkedgraph.get_root_timestamps(root_ids)

    # Find out which IDs are still current
    is_latest = client.chunkedgraph.is_latest_roots(root_ids)

    # Prepare array with death times
    deaths = np.array([dt.datetime.now(tz=dt.timezone.utc) for r in root_ids])

    # Get lineage graph for outdated root IDs
    G = client.chunkedgraph.get_lineage_graph(
        root_ids[~is_latest], timestamp_past=min(creations), as_nx_graph=True
    )

    # Get the immediate successors
    succ = np.array([next(G.successors(r)) for r in root_ids[~is_latest]])

    # Add time of death
    deaths[~is_latest] = client.chunkedgraph.get_root_timestamps(succ)

    # Find the latest creation
    latest_birth = max(creations)

    # Find the earliest death
    earliest_death = min(deaths)

    if latest_birth > earliest_death:
        raise ValueError("Given root IDs never existed at the same time.")

    return latest_birth + (earliest_death - latest_birth) / 2


@parse_neuroncriteria()
@inject_dataset(disallowed=["flat_630", "flat_571"])
def update_ids(
    id,
    stop_layer=2,
    supervoxels=None,
    timestamp=None,
    progress=True,
    *,
    dataset=None,
    **kwargs,
):
    """Retrieve the most recent version of given FlyWire (root) ID(s).

    This function works by:
        1. Check if ID is outdated (see :func:`fafbseg.flywire.is_latest_root`)
        2. If supervoxel provided, use it to update ID. Else try 3.
        3. See if we can map outdated IDs to a single up-to-date root (works
           if neuron has only seen merges). Else try 4.
        4. For uncertain IDs, fetch L2 IDs for the old root ID and the new
           candidates. Pick the candidate containing most of the original L2
           IDs.

    Parameters
    ----------
    id :            int | list-like | DataFrame | NeuronCriteria
                    Single ID or list of FlyWire (root) IDs. If DataFrame must
                    contain either a `root_id` or `root` column and optionally
                    a `supervoxel_id` or `supervoxel` column.
    stop_layer :    int
                    In case of root IDs that have been split, we need to
                    determine the most likely successor. By default we do that
                    using L2 IDs but you can speed this up by increasing the
                    stop layer.
    supervoxels :   int | list-like, optional
                    If provided will use these supervoxels to update ``id``
                    instead of sampling using the L2 IDs.
    timestamp :     int | str | datetime
                    Find root ID(s) at given date (and time). Int must be unix
                    timestamp. String must be ISO 8601 - e.g. '2021-11-15'.
                    Asking for a specific time will slow things down considerably.
    progress :      bool
                    If True, shows progress bar.
    dataset :       "public" | "production" | "sandbox", optional
                    Against which FlyWire dataset to query. If ``None`` will fall
                    back to the default dataset (see
                    :func:`~fafbseg.flywire.set_default_dataset`).

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
    :func:`~fafbseg.flywire.supervoxels_to_roots`
                    Maps supervoxels to roots. If you have supervoxel IDs for
                    your neurons this function will be significantly faster for
                    updating/mapping root IDs.

    Examples
    --------
    >>> from fafbseg import flywire
    >>> flywire.update_ids(720575940621039145)
                   old_id              new_id  confidence  changed
    0  720575940621039145  720575940631693610         1.0     True

    """
    assert stop_layer > 0, "`stop_layer` must be > 0"

    # See if we already check if this was the latest root
    is_latest = kwargs.pop("is_latest", None)

    vol = get_cloudvolume(dataset, **kwargs)

    if isinstance(id, pd.DataFrame):
        if isinstance(supervoxels, type(None)):
            if "supervoxel_id" in id.columns:
                supervoxels = id["supervoxel_id"].values
            elif "supervoxel" in id.columns:
                supervoxels = id["supervoxel"].values

        if "root_id" in id.columns:
            id = id["root_id"].values
        elif "root" in id.columns:
            id = id["root"].values
        else:
            raise ValueError(
                "DataFrame must contain either `root_id` or " "`root` column."
            )
    elif isinstance(id, pd.Series):
        id = id.values
    elif isinstance(id, pd.core.arrays.string_.StringArray):
        id = np.asarray(id)

    if isinstance(timestamp, str) and timestamp.startswith("mat"):
        client = get_cave_client(dataset=dataset)
        if timestamp == "mat" or timestamp == "mat_latest":
            timestamp = client.materialize.get_timestamp()
        else:
            # Split e.g. 'mat_432' to extract version and query timestamp
            version = int(timestamp.split("_")[1])
            timestamp = client.materialize.get_timestamp(version)

    if isinstance(id, (list, set, np.ndarray)):
        # Run is_latest once for all roots
        is_latest = is_latest_root(
            id, dataset=dataset, timestamp=timestamp, progress=progress
        )

        if isinstance(supervoxels, type(None)):
            res = [
                update_ids(
                    x,
                    dataset=dataset,
                    is_latest=il,
                    supervoxels=None,
                    timestamp=timestamp,
                    stop_layer=stop_layer,
                )
                for x, il, in navis.config.tqdm(
                    zip(id, is_latest),
                    desc="Updating",
                    leave=False,
                    total=len(id),
                    disable=not progress or len(id) == 1,
                )
            ]
            res = pd.concat(res, axis=0, sort=False, ignore_index=True)
        else:
            supervoxels = np.asarray(supervoxels)
            if len(supervoxels) != len(id):
                raise ValueError(
                    f"Number of supervoxels ({len(supervoxels)}) does "
                    f"not match number of root IDs ({len(id)})"
                )
            elif any(pd.isnull(supervoxels)):
                raise ValueError("`supervoxels` must not contain `None`")
            elif any(pd.isnull(id)):
                raise ValueError("`id` must not contain `None`")

            id = np.array(id, dtype=np.int64)

            res = pd.DataFrame()
            res["old_id"] = id
            res["new_id"] = id
            res.loc[~is_latest, "new_id"] = supervoxels_to_roots(
                supervoxels[~is_latest], timestamp=timestamp, dataset=dataset
            )
            res["conf"] = 1
            res["changed"] = res["new_id"] != res["old_id"]
        return res

    try:
        id = np.int64(id)
    except ValueError:
        raise ValueError(f'"{id} does not look like a valid root ID.')

    if id == 0 or pd.isnull(id):
        navis.config.logger.warning(
            f'Unable to update ID "{id}" - returning ' "unchanged."
        )
        return id

    # Check if outdated
    if isinstance(is_latest, type(None)):
        is_latest = is_latest_root(
            id, dataset=dataset, timestamp=timestamp, progress=progress
        )[0]

    if isinstance(timestamp, np.datetime64):
        timestamp = str(timestamp)

    if not is_latest:
        if timestamp:
            client = get_cave_client(dataset=dataset)
            get_leaves = retry(client.chunkedgraph.get_leaves)
            l2_ids_orig = get_leaves(id, stop_layer=stop_layer)

            get_roots = retry(vol.get_roots)
            roots = get_roots(l2_ids_orig, timestamp=timestamp)

            # Drop zeros
            roots = roots[roots != 0]

            if not len(roots):
                new_id = 0
                conf = 0
            else:
                uni, cnt = np.unique(roots, return_counts=True)
                new_id = uni[np.argmax(cnt)]
                conf = cnt[np.argmax(cnt)] / len(roots)
        else:
            client = get_cave_client(dataset=dataset)
            get_latest_roots = retry(client.chunkedgraph.get_latest_roots)
            # This endpoint in caveclient seems to require uint64
            pot_roots = get_latest_roots(np.uint64(id))

            # Note that we're checking whether the suggested new ID is not the same
            # as the old ID? That's because I came across a few example where the
            # lineage graph appears disconnected (e.g. 720575940613297192), perhaps
            # due to an issue in the operations log. The result of that is that
            # despite the root ID being outdated, the latest node in the graph is
            # still not the most-up-to-date ID.
            if len(pot_roots) == 1 and pot_roots[0] != id:
                new_id = pot_roots[0]
                conf = 1
            elif supervoxels:
                try:
                    supervoxels = np.int64(supervoxels)
                except ValueError:
                    raise ValueError(
                        f'"{supervoxels}" does not look like a valid ' "supervoxel ID."
                    )
                get_root_id = retry(client.chunkedgraph.get_root_id)
                new_id = get_root_id(supervoxels_to_roots)
                conf = 1
            else:
                # Get L2 IDs for the original ID
                # Note: we could also use higher level IDs
                # (stop layer 3 or 4) which would be even fasters
                get_leaves = retry(client.chunkedgraph.get_leaves)
                l2_ids_orig = get_leaves(id, stop_layer=stop_layer)
                # Get new roots for these L2 IDs
                get_roots = retry(client.chunkedgraph.get_roots)
                new_roots = get_roots(l2_ids_orig)

                # Find the most frequent new root
                roots, counts = np.unique(new_roots, return_counts=True)
                srt = np.argsort(counts)[::-1]
                roots = roots[srt]
                counts = counts[srt]

                # New ID is the most frequent ID
                new_id = roots[0]

                # Confidence is the fraction of original L2 IDs in the new ID
                conf = round(counts[0] / sum(counts), 2)
    else:
        new_id = id
        conf = 1

    return pd.DataFrame(
        [[id, new_id, conf, id != new_id]],
        columns=["old_id", "new_id", "confidence", "changed"],
    ).astype({"old_id": np.int64, "new_id": np.int64})


@inject_dataset()
def snap_to_id(
    locs,
    id,
    snap_zero=False,
    search_radius=160,
    coordinates="nm",
    max_workers=4,
    verbose=True,
    *,
    dataset=None,
):
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
    search_radius : int
                    Radius [nm] around a location to search for a position with
                    the correct ID. Lower values will be faster.
    coordinates :   "voxel" | "nm"
                    Coordinate system of `locs`. If "voxel" it is assumed to be
                    4 x 4 x 40 nm.
    max_workers :   int
    verbose :       bool
                    If True will plot summary at then end.
    dataset :       "public" | "production" | "sandbox" | "flat_630", optional
                    Against which FlyWire dataset to query. If ``None`` will fall
                    back to the default dataset (see
                    :func:`~fafbseg.flywire.set_default_dataset`).

    Returns
    -------
    (N, 3) array
                x/y/z locations that are guaranteed to map to the correct ID.

    """
    assert coordinates in ["nm", "nanometer", "nanometers", "voxel", "voxels"]

    if isinstance(locs, navis.TreeNeuron):
        locs = locs.nodes[["x", "y", "z"]].values

    # This also makes sure we work on a copy
    locs = np.array(locs, copy=True)
    assert locs.ndim == 2 and locs.shape[1] == 3

    # From hereon out we are working with nanometers
    if coordinates in ("voxel", "voxels"):
        locs *= [4, 4, 40]

    root_ids = locs_to_segments(locs, dataset=dataset, coordinates="nm")

    id_wrong = root_ids != id
    not_zero = root_ids != 0

    to_fix = id_wrong

    if not snap_zero:
        to_fix = to_fix & not_zero

    # Use parallel processes to go over the to-fix nodes
    with navis.config.tqdm(desc="Snapping", total=to_fix.sum(), leave=False) as pbar:
        with futures.ProcessPoolExecutor(max_workers=max_workers) as ex:
            loc_futures = [
                ex.submit(
                    _process_cutout,
                    id=id,
                    loc=locs[ix],
                    dataset=dataset,
                    radius=search_radius,
                )
                for ix in np.where(to_fix)[0]
            ]
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


def _process_cutout(loc, id, radius=160, dataset="production"):
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
    cutout, res, offset_nm = get_segmentation_cutout(
        bbox, dataset=dataset, root_ids=True, coordinates="nm"
    )

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


@inject_dataset()
def get_segmentation_cutout(
    bbox, root_ids=True, mip=0, coordinates="voxel", *, dataset=None
):
    """Fetch cutout of segmentation.

    Parameters
    ----------
    bbox :          array-like
                    Bounding box for the cutout::

                        [[xmin, xmax], [ymin, ymax], [zmin, zmax]]

    root_ids :      bool
                    If True, will return root IDs. If False, will return
                    supervoxel IDs. Ignored if dataset is "flat_630".
    coordinates :   "voxel" | "nm"
                    Units in which your coordinates are in. "voxel" is assumed
                    to be 4x4x40 (x/y/z) nanometers.
    dataset :       "public" | "production" | "sandbox" | "flat_630", optional
                    Against which FlyWire dataset to query. If ``None`` will fall
                    back to the default dataset (see
                    :func:`~fafbseg.flywire.set_default_dataset`).

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
    assert coordinates in ["nm", "nanometer", "nanometers", "voxel", "voxels"]

    bbox = np.asarray(bbox)
    assert bbox.ndim == 2

    if bbox.shape == (2, 3):
        pass
    elif bbox.shape == (3, 2):
        bbox = bbox.T
    else:
        raise ValueError(f"`bbox` must have shape (2, 3) or (3, 2), got {bbox.shape}")

    vol = get_cloudvolume(dataset)
    vol.mip = mip

    # First convert to nanometers
    if coordinates in ("voxel", "voxels"):
        bbox = bbox * np.array([4, 4, 40])

    # Now convert (back to) to [16, 16, 40] voxel
    bbox = (bbox / vol.scale["resolution"]).round().astype(int)

    offset_nm = bbox[0] * vol.scale["resolution"]

    # Get cutout
    cutout = vol[
        bbox[0][0] : bbox[1][0], bbox[0][1] : bbox[1][1], bbox[0][2] : bbox[1][2]
    ]

    if root_ids and ("flat" not in dataset):
        svoxels = np.unique(cutout.flatten())
        roots = supervoxels_to_roots(svoxels, dataset=vol)

        sv2r = dict(zip(svoxels[svoxels != 0], roots[svoxels != 0]))

        for k, v in sv2r.items():
            cutout[cutout == k] = v

    return cutout[:, :, :, 0], np.asarray(vol.scale["resolution"]), offset_nm


@inject_dataset(disallowed=["flat_630", "flat_571"])
def is_valid_root(x, raise_exc=False, *, dataset=None):
    """Check if ID is (potentially) valid root ID.

    Parameters
    ----------
    x :             int | str | iterable
                    ID(s) to check.
    raise_exc :     bool
                    If True and any IDs are invalid will raise an error.
                    Mostly for internal use.
    dataset :       "public" | "production" | "sandbox" | "flat_630", optional
                    Against which FlyWire dataset to query. If ``None`` will fall
                    back to the default dataset (see
                    :func:`~fafbseg.flywire.set_default_dataset`).

    Returns
    -------
    bool
            If ``x`` is a single ID.
    array
            If ``x`` is iterable.

    See Also
    --------
    :func:`~fafbseg.flywire.is_valid_supervoxel`
                    Use this function to check if a supervoxel ID is valid.

    """
    client = get_cave_client(dataset=dataset)
    vol = get_cloudvolume(client.chunkedgraph.cloudvolume_path)

    def _is_valid(x, raise_exc):
        try:
            is_valid = vol.get_chunk_layer(x) == vol.info["graph"]["n_layers"]
        except ValueError:
            is_valid = False
        
        if raise_exc and not is_valid:
            raise ValueError(f"{x} is not a valid root ID")
        
        return is_valid

    if navis.utils.is_iterable(x):
        is_valid = np.array([_is_valid(r, raise_exc=False) for r in x])
        if raise_exc and not all(is_valid):
            invalid = set(np.asarray(x)[~is_valid].tolist())
            raise ValueError(f"Invalid root IDs found: {invalid}")
        return is_valid
    else:
        return _is_valid(x, raise_exc=raise_exc)


@inject_dataset(disallowed=["flat_630", "flat_571"])
def is_valid_supervoxel(x, raise_exc=False, *, dataset=None):
    """Check if ID is (potentially) valid supervoxel ID.

    Parameters
    ----------
    x :             int | str | iterable
                    ID(s) to check.
    raise_exc :     bool
                    If True and any IDs are invalid will raise an error.
                    Mostly for internal use.
    dataset :       "public" | "production" | "sandbox" | "flat_630", optional
                    Against which FlyWire dataset to query. If ``None`` will fall
                    back to the default dataset (see
                    :func:`~fafbseg.flywire.set_default_dataset`).

    Returns
    -------
    bool
            If ``x`` is a single ID.
    array
            If ``x`` is iterable.

    See Also
    --------
    :func:`~fafbseg.flywire.is_valid_root`
                    Use this function to check if a root ID is valid.

    """
    client = get_cave_client(dataset=dataset)
    vol = get_cloudvolume(client.chunkedgraph.cloudvolume_path)

    def _is_valid(x, raise_exc):
        try:
            is_valid = vol.get_chunk_layer(x) == 1
        except ValueError:
            is_valid = False
        
        if raise_exc and not is_valid:
            raise ValueError(f"{x} is not a valid supervoxel ID")
        
        return is_valid

    if navis.utils.is_iterable(x):
        is_valid = np.array([_is_valid(r, raise_exc=False) for r in x])
        if raise_exc and not all(is_valid):
            invalid = set(np.asarray(x)[~is_valid].tolist())
            raise ValueError(f"Invalid supervoxel IDs found: {invalid}")
        return is_valid
    else:
        return _is_valid(x, raise_exc=raise_exc)


@inject_dataset(disallowed=["flat_630", "flat_571"])
def get_voxels(
    x,
    mip=0,
    sv_map=False,
    bounds=None,
    thin=False,
    progress=True,
    use_mirror=True,
    threads=4,
    *,
    dataset=None,
):
    """Fetch voxels making a up given root ID.

    Parameters
    ----------
    x :             int
                    A single root ID.
    mip :           int
                    Scale at which to fetch voxels. For example, `mip=0` is
                    at 16 x 16 x 40nm resolution. Every subsequent `mip` halves
                    the resolution.
    sv_map :        bool
                    If True, additionally return a map with the L2 ID for each
                    voxel.
    bounds :        (3, 2) or (2, 3) array, optional
                    Bounding box to return voxels in. Expected to be in 4, 4, 40
                    voxel space.
    thin :          bool
                    If True, will remove voxels at the interface of adjacent
                    supervoxels that are not supposed to be connected according
                    to the L2 graph. This is rather expensive but can help in
                    situations where a neuron self-touches.
    use_mirror :    bool
                    If True (default), will use an mirror of the base
                    segmentation for supervoxel look-up. Possibly slightly
                    slower than the production dataset but doesn't incur
                    egress charges for Princeton.
    threads :       int
                    Number of parallel threads to use for fetching the data.
    progress :      bool
                    Whether to show a progress bar or not.
    dataset :       "public" | "production" | "sandbox" | "flat_630", optional
                    Against which FlyWire dataset to query. If ``None`` will fall
                    back to the default dataset (see
                    :func:`~fafbseg.flywire.set_default_dataset`).

    Returns
    -------
    voxels :        (N, 3) np.ndarray
                    In voxel space according to `mip`.
    sv_map :        (N, ) np.ndarray
                    Supervoxel ID for each voxel. Only if `sv_map=True`.

    """
    # IDEA:
    # 1. Find surface voxels for each L2 chunk
    # 2. Get L2 graph and determine which L2 chunks are supposed to be connected
    # 3. Remove surface voxel between adjacent but not connected L2 chunks

    from .l2 import chunks_to_nm

    # This is a mirror for base segmentation
    vol = get_cloudvolume(dataset)
    client = get_cave_client()

    if use_mirror:
        sv_vol = cv.CloudVolume(
            "precomputed://https://seungdata.princeton.edu/"
            "sseung-archive/fafbv14-ws/"
            "ws_190410_FAFB_v02_ws_size_threshold_200",
            use_https=True,
            progress=False,
            fill_missing=True,
        )
    else:
        sv_vol = vol

    is_valid_root(x, raise_exc=True, dataset=dataset)

    # Get L2 chunks making up this neuron
    l2_ids = client.chunkedgraph.get_leaves(x, stop_layer=2)

    # Get supervoxels for this neuron
    sv = roots_to_supervoxels(x, dataset=dataset)[x]

    # Turn l2_ids into chunk indices
    l2_ix = [np.array(vol.mesh.meta.meta.decode_chunk_position(l)) for l in l2_ids]
    l2_ix = np.unique(l2_ix, axis=0)

    # Convert to nm
    l2_nm = np.asarray(chunks_to_nm(l2_ix, vol=vol))

    # Convert to voxel space
    l2_vxl = l2_nm // vol.meta.scales[mip]["resolution"]

    # Apply bounds
    bounds = parse_bounds(bounds)
    if not isinstance(bounds, type(None)):
        base_to_mip = np.array(vol.meta.scales[mip]["resolution"]) / [4, 4, 40]
        bounds = bounds // base_to_mip.reshape(-1, 1)
        l2_vxl = l2_vxl[np.all(l2_vxl >= bounds[:, 0], axis=1)]
        l2_vxl = l2_vxl[np.all(l2_vxl <= bounds[:, 1], axis=1)]

    voxels = []
    svids = []
    ch_size = np.array(vol.mesh.meta.meta.graph_chunk_size)
    ch_size = ch_size // (vol.mip_resolution(mip) / vol.mip_resolution(0))
    old_mip = sv_vol.mip
    old_parallel = sv_vol.parallel
    try:
        sv_vol.mip = mip
        sv_vol.parallel = threads
        for ch in tqdm(
            l2_vxl, disable=not progress, leave=False, desc="Fetching voxels"
        ):
            ct = sv_vol[
                ch[0] : ch[0] + ch_size[0],
                ch[1] : ch[1] + ch_size[1],
                ch[2] : ch[2] + ch_size[2],
            ][:, :, :, 0]
            is_root = np.isin(ct, sv)
            this_vxl = np.dstack(np.where(is_root))[0]
            this_vxl = this_vxl + ch
            voxels.append(this_vxl)

            if sv_map or thin:
                svids.append(ct[is_root])
    except BaseException:
        raise
    finally:
        sv_vol.mip = old_mip
        sv_vol.parallel = old_parallel

    # uint 16 should be sufficient because even at mip 0 the volume has
    # shape (54100, 28160, 7046) -> doesn't exceed 65_535
    voxels = np.vstack(voxels).astype("uint16")
    if len(svids):
        svids = np.concatenate(svids)

    if thin:
        from .l2 import get_l2_graph

        try:
            from pykdtree.kdtree import KDTree
        except ImportError:
            from scipy.spatial import cKDTree as KDTree

        # Get the l2 ID for each supervoxel
        l2_ids = vol.get_roots(svids, stop_layer=2)
        l2_dict = dict(zip(svids, l2_ids))

        # Get the l2 graph
        G = get_l2_graph(x)

        # Create KD tree for all voxels
        tree = KDTree(voxels)

        # Create a mask for invalidated voxels
        invalid = np.zeros(len(voxels), dtype=bool)

        # Now go over each supervoxel
        for sv in tqdm(
            np.unique(svids), disable=not progress, desc="Thinning", leave=False
        ):
            # Get the voxels for this supervoxel
            is_this_sv = svids == sv

            # If supervoxel has no voxels just continue
            if not np.any(is_this_sv):
                continue

            # Get all supervoxels that could be connected to this supervoxel
            is_this_l2 = l2_ids == l2_dict[sv]
            is_connected_l2 = np.isin(l2_ids, list(G.neighbors(l2_dict[sv])))

            # The mask needs to exclude anything that:
            # Isn't this supervoxel OR is supposed to be connected OR
            # has already been invalidated in a prior run
            mask = is_this_l2 | is_connected_l2 | invalid

            # Find "other" voxels that touch voxels for this supervoxel
            dist, ix = tree.query(
                voxels[is_this_sv], mask=mask, distance_upper_bound=1.75
            )
            is_touching = dist < np.inf

            if not np.any(is_touching):
                continue

            invalid[np.where(is_this_sv)[0][is_touching]] = True

        voxels = voxels[~invalid]
        svids = svids[~invalid]

    if not sv_map:
        return voxels
    else:
        return voxels, svids
