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
           'get_cave_client', 'get_neuropil_volumes', 'get_lr_position']

FLYWIRE_DATASETS = {'production': 'fly_v31',
                    'sandbox': 'fly_v26'}

CAVE_DATASETS = {'production': 'flywire_fafb_production',
                 'sandbox': 'flywire_fafb_sandbox'}

# Initialize without a volume
fw_vol = None
cave_clients = {}

# Data stuff
fp = Path(__file__).parent
data_path = fp.parent / 'data'
area_ids = None
vol_names = None


def get_neuropil_volumes(neuropils):
    """Load FlyWire neuropil volumes.

    These meshes were originally created by for the JFRC2 brain template
    (for citation and details see 10.5281/zenodo.10567). Here, we transformed
    them to FlyWire (FAFB14.1) space.

    Parameters
    ----------
    neuropils :     str | list thereof
                    Neuropil name(s) - e.g. 'LH_R' or ['LH_R', 'LH_L'].

    Returns
    -------
    meshes :        single navis.Volume or list thereof

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

            raise ValueError(f'No mesh for neuropil "{neuropils}". Available '
                             f'neuropils: {", ".join(available)}')

        f = zip.read(f'{neuropils}.stl')
        m = tm.load_mesh(BytesIO(f), file_type='stl')

    return navis.Volume(m, name=neuropils)


def get_synapse_areas(ind):
    """Lazy-load synapse areas.

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


def get_cave_client(dataset='production', token=None, check_stale=True,
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
            if (dt.datetime.now() - client.birth_day) > dt.timedelta(minutes=30):
                force_new = True

    if datastack not in cave_clients or force_new:
        cave_clients[datastack] = CAVEclient(datastack, auth_token=token)
        cave_clients[datastack].birth_day = dt.datetime.now()

    return cave_clients[datastack]


def get_chunkedgraph_secret(domain='prod.flywire-daf.com'):
    """Get chunked graph secret.

    Parameters
    ----------
    domain :    str
                Domain to get the secret for. Only relevant for
                ``cloudvolume>=3.11.0``.

    Returns
    -------
    token :     str

    """
    if hasattr(cv.secrets, 'cave_credentials'):
        token = cv.secrets.cave_credentials(domain).get('token', None)
        if not token:
            raise ValueError(f'No chunkedgraph secret for domain {domain} '
                             'found. Please see '
                             'fafbseg.flywire.set_chunkedgraph_secret to set '
                             'your secret.')
    else:
        try:
            token = cv.secrets.chunkedgraph_credentials['token']
        except BaseException:
            raise ValueError('No chunkedgraph secret found. Please see '
                             'fafbseg.flywire.set_chunkedgraph_secret to set your '
                             'secret.')
    return token


def set_chunkedgraph_secret(token, filepath=None,
                            domain='prod.flywire-daf.com'):
    """Set chunked graph secret (called "cave credentials" now).

    Parameters
    ----------
    token :     str
                Get your token from
                https://globalv1.flywire-daf.com/auth/api/v1/refresh_token
    filepath :  str filepath
                Path to secret file. If not provided will store in default path:
                ``~/.cloudvolume/secrets/{domain}-cave-secret.json``
    domain :    str
                The domain (incl subdomain) this secret is for.

    """
    assert isinstance(token, str), f'Token must be string, got "{type(token)}"'

    if not filepath:
        filepath = f'~/.cloudvolume/secrets/{domain}-cave-secret.json'
    elif not filepath.endswith('/chunkedgraph-secret.json'):
        filepath = os.path.join(filepath, f'{domain}-cave-secret.json')
    elif not filepath.endswith('.json'):
        filepath = f'{filepath}.json'

    filepath = Path(filepath).expanduser()

    # Make sure this file (and the path!) actually exist
    if not filepath.exists():
        if not filepath.parent.exists():
            filepath.parent.mkdir(parents=True)
        filepath.touch()

    with open(filepath, 'w+') as f:
        json.dump({'token': token}, f)

    # We need to reload cloudvolume for changes to take effect
    reload(cv.secrets)
    reload(cv)

    # Should also reset the volume after setting the secret
    global fw_vol
    fw_vol = None

    print("Token succesfully stored in ", filepath)


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


def parse_volume(vol, **kwargs):
    """Parse CloudVolume."""
    global fw_vol
    if 'CloudVolume' not in str(type(vol)):
        if not isinstance(vol, str):
            raise ValueError(f'Unable to initialize CloudVolume from "{type(vol)}"')

        if not utils.is_url(vol):
            # We are assuming this is the dataset
            # Map "production" and "sandbox" with to their correct designations
            vol = FLYWIRE_DATASETS.get(vol, vol)

            # Below is supposedly the "old" api (/1.0/)
            # vol = f'graphene://https://prodv1.flywire-daf.com/segmentation/1.0/{vol}'

            # This is the new url
            vol = f'graphene://https://prod.flywire-daf.com/segmentation/table/{vol}'

            # This might eventually become the new url
            # vol = f'graphene://https://prodv1.flywire-daf.com/segmentation_proc/table/{vol}'

        if not vol.startswith('graphene://'):
            vol = f'graphene://{vol}'

        #  Change default volume if necessary
        if not fw_vol or getattr(fw_vol, 'path', None) != vol:
            # Set and update defaults from kwargs
            defaults = dict(mip=0,
                            fill_missing=True,
                            use_https=True,  # this way google secret is not needed
                            progress=False)
            defaults.update(kwargs)

            # Check if chunkedgraph secret exists
            secret = os.path.expanduser('~/.cloudvolume/secrets/chunkedgraph-secret.json')
            if not os.path.isfile(secret):
                # If not secrets but environment variable use this
                if 'CHUNKEDGRAPH_SECRET' in os.environ and 'secrets' not in defaults:
                    defaults['secrets'] = {'token': os.environ['CHUNKEDGRAPH_SECRET']}

            fw_vol = cv.CloudVolume(vol, **defaults)
            fw_vol.path = vol
    else:
        fw_vol = vol
    return fw_vol


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
                except requests.HTTPError:
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
                           allowed_values=('nm', 'nanometers', 'nanometers',
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
