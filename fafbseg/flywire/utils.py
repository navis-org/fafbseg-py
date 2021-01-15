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

import json
import os

from pathlib import Path
from importlib import reload

import cloudvolume as cv

from .. import utils


__all__ = ['set_chunkedgraph_secret']

FLYWIRE_DATASETS = {'production': 'fly_v31',
                    'sandbox': 'fly_v26'}


def set_chunkedgraph_secret(token, filepath=None):
    """Set chunked graph secret.

    Parameters
    ----------
    token :     str
                Get your token from
                https://globalv1.flywire-daf.com/auth/api/v1/refresh_token
    filepath :  str filepath
                Path to secret file. If not provided will store in default path:
                ``~/.cloudvolume/secrets/chunkedgraph-secret.json``

    """
    assert isinstance(token, str), f'Token must be string, got "{type(token)}"'

    if not filepath:
        filepath = '~/.cloudvolume/secrets/chunkedgraph-secret.json'
    elif not filepath.endswith('/chunkedgraph-secret.json'):
        filepath = f'{filepath}/chunkedgraph-secret.json'
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
            # This is the new url (/table/)
            vol = f'graphene://https://prodv1.flywire-daf.com/segmentation/table/{vol}'

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


# Initialize without a volume
fw_vol = None
