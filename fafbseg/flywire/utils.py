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
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU General Public License for more details.

import functools
import json
import navis
import os
import pytz
import time
import requests
import warnings

from caveclient import CAVEclient
from pathlib import Path
from importlib import reload
from zipfile import ZipFile
from io import BytesIO

import cloudvolume as cv
import datetime as dt
import trimesh as tm
import numpy as np
import pandas as pd

from .. import utils


__all__ = ['set_chunkedgraph_secret', 'get_chunkedgraph_secret',
           'get_cave_client', 'get_neuropil_volumes', 'get_lr_position',
           'set_default_dataset', 'find_mat_version']

FLYWIRE_DATASETS = {'production': 'fly_v31',
                    'sandbox': 'fly_v26',
                    'public': 'flywire_public'}

FLYWIRE_URLS = {'production': 'graphene://https://prod.flywire-daf.com/segmentation/1.0/fly_v31',
                'sandbox': 'graphene://https://prod.flywire-daf.com/segmentation/1.0/fly_v26',
                'public': 'graphene://https://prodv1.flywire-daf.com/segmentation/1.0/flywire_public',
                'flat_630': 'precomputed://gs://flywire_v141_m630',
                'flat_571': 'precomputed://gs://flywire_v141_m526',
                'flat_783': 'precomputed://gs://flywire_v141_m783'}

CAVE_DATASETS = {'production': 'flywire_fafb_production',
                 'flat_783': 'flywire_fafb_production',
                 'flat_630': 'flywire_fafb_public',
                 'flat_571': 'flywire_fafb_production',
                 'sandbox': 'flywire_fafb_sandbox',
                 'public': 'flywire_fafb_public'}

SILENCE_FIND_MAT_VERSION = False

# Initialize without a volume
cloud_volumes = {}
cave_clients = {}

# Data stuff
fp = Path(__file__).parent
data_path = fp.parent / 'data'
area_ids = None
vol_names = None

# The default dataset
DEFAULT_DATASET = os.environ.get('FLYWIRE_DEFAULT_DATASET', 'public')

# Some useful data types
INT_DTYPES = (np.int32, np.int64, int, np.uint32, np.uint64)
FLOAT_DTYPES = (np.float32, np.float64, float)
STR_DTYPES = (str, np.str_)


def match_dtype(x, target_dt):
    """Make sure that input has same dtype as target.

    This function only maches the broad data type (float, integer, string), not
    e.g. the exact precision.

    Parameters
    ----------
    x
                Input to be converted.
    target_dt
                The target data type.

    Returns
    -------
    x :
                Input with matching dtype. Lists and tuples will be converted
                to numpy arrays.

    """
    if isinstance(x, (list, tuple, np.ndarray)):
        x = np.asarray(x)
        if target_dt in INT_DTYPES:
            x = x.astype(np.int64)
        elif target_dt in FLOAT_DTYPES:
            x = x.astype(np.float64)
        elif target_dt in STR_DTYPES:
            x = x.astype(str)
    else:
        if target_dt in INT_DTYPES:
            x = np.int64(x)
        elif target_dt in FLOAT_DTYPES:
            x = float(x)
        elif target_dt in STR_DTYPES:
            x = str(x)

    return x


def set_default_dataset(dataset):
    """Set the default FlyWire dataset for this session.

    Alternatively, you can also use a FLYWIRE_DEFAULT_DATASET environment
    variable (must be set before starting Python).

    Parameters
    ----------
    dataset :   "production" | "public" | "sandbox" | "flat_630"
                Dataset to be used by default.

    Examples
    --------
    >>> from fafbseg import flywire
    >>> flywire.set_default_dataset('public')
    Default dataset set to "public".

    """
    if dataset not in FLYWIRE_URLS and dataset not in get_cave_datastacks():
        datasets = np.unique(list(FLYWIRE_URLS) + get_cave_datastacks())
        raise ValueError(f'`dataset` must be one of: {", ".join(datasets)}.')

    global DEFAULT_DATASET
    DEFAULT_DATASET = dataset
    print(f'Default dataset set to "{dataset}".')


def inject_dataset(allowed=None, disallowed=None):
    """Inject current default dataset."""
    if isinstance(allowed, str):
        allowed = [allowed]
    if isinstance(disallowed, str):
        disallowed = [disallowed]
    def outer(func):
        @functools.wraps(func)
        def inner(*args, **kwargs):
            if kwargs.get('dataset', None) is None:
                kwargs['dataset'] = DEFAULT_DATASET

            ds = kwargs['dataset']
            if allowed and ds not in allowed:
                raise ValueError(f'Dataset "{ds}" not allowed for function {func}. '
                                 f'Accepted datasets: {allowed}')
            if disallowed and ds in disallowed:
                raise ValueError(f'Dataset "{ds}" not allowed for function {func}.')
            return func(*args, **kwargs)
        return inner
    return outer


def get_neuropil_volumes(neuropils):
    """Load FlyWire neuropil volumes.

    These meshes were originally created by for the JFRC2 brain template
    (for citation and details see 10.5281/zenodo.10567). Here, we transformed
    them to FlyWire (FAFB14.1) space.

    Parameters
    ----------
    neuropils :     str | list thereof | None
                    Neuropil name(s) - e.g. 'LH_R' or ['LH_R', 'LH_L']. Use
                    ``None`` to get an array of available neuropils.

    Returns
    -------
    meshes :        single navis.Volume or list thereof

    Examples
    --------
    Load a single volume:

    >>> from fafbseg import flywire
    >>> al_r = flywire.get_neuropil_volumes('AL_R')
    >>> al_r
    <navis.Volume(name=AL_R, color=(0.85, 0.85, 0.85, 0.2), vertices.shape=(622, 3), faces.shape=(1240, 3))>

    Load multiple volumes:

    >>> from fafbseg import flywire
    >>> al_lr = flywire.get_neuropil_volumes(['AL_R', 'AL_L'])
    >>> al_lr
    [<navis.Volume(name=AL_R, color=(0.85, 0.85, 0.85, 0.2), vertices.shape=(622, 3), faces.shape=(1240, 3))>,
     <navis.Volume(name=AL_L, color=(0.85, 0.85, 0.85, 0.2), vertices.shape=(612, 3), faces.shape=(1228, 3))>]

    Get a list of available volumes:

     >>> from fafbseg import flywire
     >>> available = flywire.get_neuropil_volumes(None)

    """
    if navis.utils.is_iterable(neuropils):
        return [get_neuropil_volumes(n) for n in neuropils]

    with ZipFile(data_path / 'JFRC2NP.surf.fw.zip', 'r') as zip:
        try:
            f = zip.read(f'{neuropils}.stl')
        except KeyError:
            available = []
            for file in zip.filelist:
                fname = file.filename.split('/')[-1]
                if not fname.endswith('.stl') or fname.startswith('.'):
                    continue
                available.append(fname.replace('.stl', ''))
            available = sorted(available)

            if neuropils:
                raise ValueError(f'No mesh for neuropil "{neuropils}". Available '
                                 f'neuropils: {", ".join(available)}')
            else:
                return np.array(available)

        f = zip.read(f'{neuropils}.stl')
        m = tm.load_mesh(BytesIO(f), file_type='stl')

    return navis.Volume(m, name=neuropils)


def get_synapse_areas(ind):
    """Lazy-load synapse areas (neuropils).

    Parameters
    ----------
    ind :       (N, ) iterable
                Synapse indices (shows up as `id` in synapse table).

    Returns
    -------
    areas :     (N, ) array
                Array with neuropil name for each synapse. Unassigned synapses
                come back as "NA".

    """
    global area_ids, vol_names

    if isinstance(area_ids, type(None)):
        area_ids = np.load(data_path / 'global_area_ids.npy.zip')['global_area_ids']
        with open(data_path / 'volume_name_dict.json') as f:
            vol_names = json.load(f)
            vol_names = {int(k): v for k, v in vol_names.items()}
            vol_names[-1] = 'NA'

    return np.array([vol_names[i] for i in area_ids[ind]])


@functools.lru_cache
def get_cave_datastacks():
    """Get available CAVE datastacks."""
    return CAVEclient().info.get_datastacks()


@functools.lru_cache
def get_datastack_segmentation_source(datastack):
    """Get segmentation source for given CAVE datastack."""
    return CAVEclient().info.get_datastack_info(datastack_name=datastack)['segmentation_source']


@inject_dataset()
def get_cave_client(*, dataset=None, token=None, check_stale=True,
                    force_new=False):
    """Get CAVE client.

    Currently, the CAVE client pulls the available materialization versions
    ONCE on initialization. This means that if the same client is used for over
    24h it will be unaware of any new materialization versions which will slow
    down live queries substantially. We try to detect whether the client may
    have gone stale but this may not always work perfectly.

    Parameters
    ----------
    dataset :       str
                    Data set to create client for.
    token :         str, optional
                    Your chunked graph secret (i.e. "CAVE secret"). If not
                    provided will try reading via cloud-volume.
    check_stale :   bool
                    Check if any existing client has gone stale. Currently, we
                    check if the cached materialization meta data needs
                    refreshing and we automatically refresh the client every
                    hour.
    force_new :     bool
                    If True, we force a re-initialization.

    Returns
    -------
    CAVEclient

    """
    if not token:
        token = get_chunkedgraph_secret()

    datastack = CAVE_DATASETS.get(dataset, dataset)

    if datastack in cave_clients and not force_new and check_stale:
        # Get the existing client
        client = cave_clients[datastack]
        # Get the (likely cached) materialization meta data
        mds = client.materialize.get_versions_metadata()
        # Check if any of the versions are expired
        now = pytz.UTC.localize(dt.datetime.utcnow())
        for v in mds:
            if v['expires_on'] <= now:
                force_new = True
                break

        # Over the weekend no new versions are materialized. The last version
        # from Friday will persist into middle of the next week - i.e. not
        # expire on Monday. Therefore, on Mondays only, we will also
        # force an update if the client is older than 30 minutes
        if now.weekday() in (0, ) and not force_new:
            if (dt.datetime.now() - client._created_at) > dt.timedelta(minutes=30):
                force_new = True

    if datastack not in cave_clients or force_new:
        cave_clients[datastack] = CAVEclient(datastack, auth_token=token)
        cave_clients[datastack]._created_at = dt.datetime.now()

    return cave_clients[datastack]


def get_chunkedgraph_secret(domain=('global.daf-apis.com', 'prod.flywire-daf.com')):
    """Get local FlyWire chunkedgraph/CAVE secret.

    Parameters
    ----------
    domain :    str | list thereof
                Domain to get the secret for.

    Returns
    -------
    token :     str

    """
    if isinstance(domain, str):
        domain = [domain]

    token = None
    for dom in domain:
        token = cv.secrets.cave_credentials(dom).get('token', None)
        if token:
            break

    if not token:
        raise ValueError(f'No chunkedgraph/CAVE secret for domain(s) {domain} '
                        'found. Please see fafbseg.flywire.set_chunkedgraph_secret '
                        'to store your API token.')
    return token


def set_chunkedgraph_secret(token, overwrite=False, **kwargs):
    """Set FlyWire chunkedgraph/CAVE secret.

    This is just a thin wrapper around ``caveclient.CAVEclient.auth.save_token()``.

    Parameters
    ----------
    token :     str
                Get your token from
                https://global.daf-apis.com/auth/api/v1/user/token. If that URL
                returns an empty list ``[]`` you should visit
                https://global.daf-apis.com/auth/api/v1/create_token instead.
    overwrite : bool
                Whether to overwrite any existing secret.
    **kwargs
                Keyword arguments are passed through to
                ``caveclient.CAVEclient.save_token()``.

    """
    assert isinstance(token, str), f'Token must be string, got "{type(token)}"'

    # Save token
    CAVEclient().auth.save_token(token, overwrite=overwrite, **kwargs)

    # We need to reload cloudvolume for changes to take effect
    reload(cv.secrets)
    reload(cv)

    # Should also reset the volume after setting the secret
    global fw_vol
    fw_vol = None

    print("Token succesfully stored.")


def parse_root_ids(x):
    """Parse root IDs.

    Always returns an array of integers.
    """
    if isinstance(x, navis.BaseNeuron):
        ids = [x.id]
    elif isinstance(x, navis.NeuronList):
        ids = x.id
    elif isinstance(x, (int, np.int64)):
        ids = [x]
    else:
        ids = utils.make_iterable(x, force_type=np.int64)

    # Make sure we are working with proper numerical IDs
    try:
        return np.asarray(ids, dtype=np.int64)
    except ValueError:
        raise ValueError(f'Unable to convert given root IDs to integer: {ids}')
    except BaseException:
        raise


def get_cloudvolume(dataset, **kwargs):
    """Get CloudVolume for given dataset."""
    # If this already is a CloudVolume just pass it through
    if "CloudVolume" in  str(type(dataset)):
        return dataset
    else:
        if not isinstance(dataset, str):
            raise ValueError(f'Unable to initialize CloudVolume from "{type(dataset)}"')

        # Translate into a URL
        if not utils.is_url(dataset):
            # Map "production" and "sandbox" to their URLs
            if dataset in FLYWIRE_URLS:
                dataset = FLYWIRE_URLS[dataset]
            # Failing that, see if CAVE knows about them
            elif dataset in get_cave_datastacks():
                dataset = get_datastack_segmentation_source(dataset)
            # Otherwise we will assume that this already is a segmentation URL

        # Add this volume if it does not already exists
        if dataset not in cloud_volumes:
            # Set and update defaults from kwargs
            defaults = dict(mip=0,
                            fill_missing=True,
                            cache=False,
                            use_https=True,  # this way google secret is not needed
                            progress=False)
            defaults.update(kwargs)

            # Check if chunkedgraph secret exists
            # This probably needs yanking!
            #secret = os.path.expanduser('~/.cloudvolume/secrets/chunkedgraph-secret.json')
            #if not os.path.isfile(secret):
            #    # If not secrets but environment variable use this
            #    if 'CHUNKEDGRAPH_SECRET' in os.environ and 'secrets' not in defaults:
            #        defaults['secrets'] = {'token': os.environ['CHUNKEDGRAPH_SECRET']}

            cloud_volumes[dataset] = cv.CloudVolume(dataset, **defaults)
            cloud_volumes[dataset].path = dataset

        return cloud_volumes[dataset]


def retry(func, retries=5, cooldown=2):
    """Retry function on HTTPError.

    This also suppresses UserWarnings (because we typically use this for stuff
    like the l2 Cache).

    Parameters
    ----------
    cooldown :  int | float
                Cooldown period in seconds between attempts.
    retries :   int
                Number of retries before we give up. Every subsequent retry
                will delay by an additional `retry`.

    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        for i in range(1, retries + 1):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    return func(*args, **kwargs)
                except KeyboardInterrupt:
                    raise
                except requests.RequestException:
                    if i >= retries:
                        raise
                except BaseException:
                    raise
                time.sleep(cooldown * i)
    return wrapper


def parse_bounds(x):
    """Parse bounds.

    Parameters
    ----------
    x :     (3, 2) array | (2, 3) array | None

    Returns
    -------
    bounds :    (3, 2) np.array

    """
    if isinstance(x, type(None)):
        return x

    x = np.asarray(x)

    if not x.ndim == 2 or x.shape not in [(3, 2), (2, 3)]:
        raise ValueError('Must provide bounding box as (3, 2) or (2, 3) array, '
                         f'got {x.shape}')

    if x.shape == (2, 3):
        x = x.T

    return np.vstack((x.min(axis=1), x.max(axis=1))).T


def get_lr_position(x, coordinates='nm'):
    """Find out if given xyz positions are on the fly's left or right.

    This works by:
     1. Mirror positions from one side to the other (requires `flybrains`)
     2. Substracting original from the mirrored x-coordinate

    Parameters
    ----------
    x :             (N, 3) array | TreeNeuron | MeshNeuron | Dotprops
                    Array of xyz coordinates or a neuron. If a navis neuron,
                    will use nodes, vertex or point coordinates for TreeNeurons,
                    MeshNeurons and Dotprops, respectively.
    coordinates :   "nm" | "voxel"
                    Whether coordinates are in nm or voxel space.

    Returns
    -------
    xm :            (N, ) array
                    A vector of point displacements in nanometers where 0 is
                    at the midline and positive values are to the fly's right.

    Examples
    --------
    >>> from fafbseg import flywire
    >>> # Three example points: right, left, ~center
    >>> flywire.get_lr_position([[104904, 47464, 5461],
    ...                          [140648, 49064, 2262],
    ...                          [131256, 29984, 2358]],
    ...                         coordinates='voxel')
    array([110501.5, -39480. ,    306.5])

    """
    try:
        import flybrains
    except ImportError:
        raise ImportError('This function requires `flybrains` to be '
                          'installed:\n pip3 install flybrains')

    # The FlyWire mirror registration is only part of the most recent version
    try:
        _ = navis.transforms.registry.find_template('FLYWIRE')
    except ValueError:
        raise ImportError('Looks like your version of `flybrains` is outdated. '
                          'Please update:\n pip3 install flybrains -U')

    navis.utils.eval_param(coordinates, name='coordinates',
                           allowed_values=('nm', 'nanometer', 'nanometers',
                                           'voxel', 'voxels'))

    if navis.utils.is_iterable(x):
        x = np.asarray(x)
    elif isinstance(x, pd.DataFrame):
        if x.shape[1] == 3:
            x = x.values
        elif all([c in x.columns for c in ['x', 'y', 'z']]):
            x = x[['x', 'y', 'z']].values
    elif isinstance(x, navis.TreeNeuron):
        x = x.nodes[['x', 'y', 'z']].values
    elif isinstance(x, navis.MeshNeuron):
        x = x.vertices
    elif isinstance(x, navis.Dotprops):
        x = x.points

    if not isinstance(x, np.ndarray):
        raise TypeError(f'Expected numpy array or neuron, got "{type(x)}"')
    elif x.ndim != 2 or x.shape[1] != 3:
        raise TypeError(f'Expected (N, 3) numpy array, got {x.shape}')

    # Scale if required
    if coordinates in ('voxel', 'voxels'):
        x = x * [4, 4, 40]

    # Mirror -> this should be using the landmark-based transform in flybrains
    m = navis.mirror_brain(x, template='FLYWIRE')

    return (m[:, 0] - x[:, 0]) / 2


@inject_dataset()
def find_mat_version(ids,
                     verbose=True,
                     allow_multiple=False,
                     raise_missing=True,
                     dataset=None):
    """Find a materialization version (or live) for given IDs.

    Parameters
    ----------
    ids :           iterable
                    Root IDs to check.
    verbose :       bool
                    Whether to print results of search. See also the
                    `flywire.utils.silence_find_mat_version` context manager to
                    silence output.
    allow_multiple : bool
                    If True, will track if IDs can be found spread across multiple
                    materialization versions if there is no single one containing
                    all.
    raise_missing : bool
                    Only relevant if `allow_multiple=True`. If False, will return
                    versions even if some IDs could not be found.

    Returns
    -------
    version :       int | "live"
                    A single version (including "live") that contains all given
                    root IDs.
    versions :      np.ndarray
                    If no single version was found and `allow_multiple=True` will
                    return a vector of `len(ids)` with the latest version at which
                    the respective ID can be found.
                    Important: "live" version will be return as -1!
                    If `raise_missing=False` and one or more root IDs could not
                    be found in any of the available materialization versions
                    these IDs will be return as version 0.

    """
    # If dataset is the flat segmentation we can take a shortcut
    if dataset == 'flat_630':
        return 630
    elif dataset == 'flat_571':
        return 571

    ids = np.asarray(ids)

    client = get_cave_client(dataset=dataset)

    # For each ID track the most recent valid version
    latest_valid = np.zeros(len(ids), dtype=np.int32)

    # Go over each version (start with the most recent)
    for i, version in enumerate(sorted(client.materialize.get_versions(), reverse=True)):
        ts_m = client.materialize.get_timestamp(version)

        # Check which root IDs were valid at the time
        is_valid = client.chunkedgraph.is_latest_roots(ids, timestamp=ts_m)

        # Update latest valid versions
        latest_valid[(latest_valid == 0) & is_valid] = version

        if all(is_valid):
            if verbose and not SILENCE_FIND_MAT_VERSION:
                print(f'Using materialization version {version}.')
            return version

    # If no single materialized version can be found, see if we can get
    # by with the live materialization
    is_latest = client.chunkedgraph.is_latest_roots(ids, timestamp=None)
    latest_valid[(latest_valid == 0) & is_latest] = -1  # track "live" as -1
    if all(is_latest) and dataset != 'public':  # public does not have live
        if verbose and not SILENCE_FIND_MAT_VERSION:
            print('Using live materialization')
        return 'live'

    if allow_multiple and any(latest_valid != 0):
        if all(latest_valid != 0):
            if verbose and not SILENCE_FIND_MAT_VERSION:
                print(f"Found root IDs spread across {len(np.unique(latest_valid))} "
                      "materialization versions.")
            return latest_valid
        
        msg = (f"Found root IDs spread across {len(np.unique(latest_valid)) - 1} "
               f"materialization versions but {(latest_valid == 0).sum()} IDs "
               "do not exist in any of the materialized tables.")

        if not raise_missing:
            if verbose and not SILENCE_FIND_MAT_VERSION:
                print(msg)
            return latest_valid
        else:
            raise MaterializationMatchError(msg)

    if dataset not in ('public, '):
        raise MaterializationMatchError(
            'Given root IDs do not (co-)exist in any of the available '
            'materialization versions (including live). Try updating '
            'root IDs and rerun your query.')
    else:
        raise MaterializationMatchError(
            'Given root IDs do not (co-)exist in any of the available '
            'public materialization versions. Please make sure that '
            'the root IDs do exist and rerun your query.')


def _is_valid_version(ids, version, dataset):
    """Test if materialization version is valid for given root IDs."""
    client = get_cave_client(dataset=dataset)

    # If this is not even a valid version (for this dataset) return False
    if version not in client.materialize.get_versions():
        return False

    ts_m = client.materialize.get_timestamp(version)

    # Check which root IDs were valid at the time
    is_valid = client.chunkedgraph.is_latest_roots(ids, timestamp=ts_m)

    if all(is_valid):
        return True

    return False


def package_timestamp(timestamp, name="timestamp"):
    # Copied from caveclient
    if timestamp is None:
        query_d = {}
    else:
        if timestamp.tzinfo is None:
            timestamp = pytz.UTC.localize(timestamp)
        else:
            timestamp = timestamp.astimezone(dt.timezone.utc)

        query_d = {name: timestamp.timestamp()}
    return query_d


class silence_find_mat_version:
    def __enter__(self):
        global SILENCE_FIND_MAT_VERSION
        SILENCE_FIND_MAT_VERSION = True

    def __exit__(self, exc_type, exc_value, exc_tb):
        global SILENCE_FIND_MAT_VERSION
        SILENCE_FIND_MAT_VERSION = False


class MaterializationMatchError(Exception):
    pass
