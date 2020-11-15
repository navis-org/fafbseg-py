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

import requests
import cloudvolume as cv
from urllib.parse import urlparse, parse_qs

__all__ = ['decode_ngl_url', 'generate_open_ends_url']


def decode_ngl_url(url, ret='brief'):
    """Decode neuroglancer URL.

    Parameters
    ----------
    url :       str
                URL to decode. Can be shortened URL.
    ret :       "brief" | "full"
                If "brief", will only return "position" (in voxels), "selected"
                segment IDs and "annotations". If full, will return entire scene.

    Returns
    -------
    dict

    Examples
    --------
    >>> from fafbseg import flywire
    >>> flywire.decode_ngl_url('https://ngl.flywire.ai/?json_url=https://globalv1.flywire-daf.com/nglstate/6267328375291904')
    {'position': [132715.625, 55805.6796875, 3289.61181640625],
     'annotations': [],
     'selected': ['720575940621039145']}

    """
    assert isinstance(url, (str, dict))
    assert ret in ['brief', 'full']

    query = parse_qs(urlparse(url).query, keep_blank_values=True)

    if 'json_url' in query:
        # Fetch state
        token = cv.secrets.chunkedgraph_credentials['token']
        r = requests.get(query['json_url'][0], headers={'Authorization': f"Bearer {token}"})
        r.raise_for_status()

        scene = r.json()
    else:
        scene = query

    if ret == 'brief':
        seg_layers = [l for l in scene['layers'] if l.get('type') == 'segmentation_with_graph']
        an_layers = [l for l in scene['layers'] if l.get('type') == 'annotation']
        return {'position': scene['navigation']['pose']['position']['voxelCoordinates'],
                'annotations': [a for l in an_layers for a in l.get('annotations', [])],
                'selected': [s for l in seg_layers for s in l.get('segments', [])]}

    return scene


def generate_open_ends_url(x):
    """Generate a flywire URL with potential open ends for given neuron.

    Parameters
    ----------
    x :     flywire ID | navis.TreeNeuron | mesh
            ID of neuron to generate open ends for.

    """
    pass
