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
import copy
import json
import navis
import pymaid
import pyperclip
import requests
import uuid
import webbrowser

import matplotlib.colors as mcl
import numpy as np
import pandas as pd

from urllib.parse import urlparse, parse_qs, quote

from . import utils

__all__ = ['decode_url', 'encode_url']

NGL_URL = 'https://ngl.flywire.ai'
MINIMAL_SCENE = {'layers': [{'source': 'precomputed://gs://microns-seunglab/drosophila_v0/alignment/image_rechunked',
                             'type': 'image',
                             'blend': 'default',
                             'shaderControls': {},
                             'name': 'Production-image'
                             },
                            {'source': 'graphene://https://prod.flywire-daf.com/segmentation/1.0/{dataset}',
                             'type': 'segmentation_with_graph',
                             'selectedAlpha': 0.14,
                             'segments': [],
                             'skeletonRendering': {'mode2d': 'lines_and_points', 'mode3d': 'lines'},
                             'graphOperationMarker': [{'annotations': [], 'tags': []},
                                                      {'annotations': [], 'tags': []}],
                             'pathFinder': {'color': '#ffff00',
                                            'pathObject': {'annotationPath': {'annotations': [], 'tags': []},
                                                           'hasPath': False}
                                            },
                             'name': 'Production-segmentation_with_graph'}],
                 'navigation': {'pose': {'position': {'voxelSize': [4, 4, 40],
                                                      'voxelCoordinates': [118073, 57192, 4070]}},  # default
                                'zoomFactor': 2.8},  # Zoom in 2d
                 'perspectiveOrientation': [0, 0, 0, 1],  # This is a frontal perspective in 3d
                 'perspectiveZoom': 21000,  # Zoom in 3d
                 'jsonStateServer': 'https://globalv1.flywire-daf.com/nglstate/post',
                 'selectedLayer': {'layer': 'Production-segmentation_with_graph',
                                   'visible': True},
                 'layout': 'xy-3d'}
STATE_URL = "https://globalv1.flywire-daf.com/nglstate"
session = None


def encode_url(segments=None, annotations=None, coords=None, skeletons=None,
               seg_colors=None, invis_segs=None, dataset='production', scene=None,
               open_browser=False, to_clipboard=False, short=True):
    """Encode data as FlyWire neuroglancer scene.

    Parameters
    ----------
    segments :      int | list of int, optional
                    Segment IDs to have selected.
    annotations :   (N, 3) array, optional
                    2d of xyz coordinates that will be added as annotation
                    layer. If you need more control over this, see
                    :func:`fafbseg.flywire.add_annotation_layer`.
    coords :        (3, ) array, optional
                    (X, Y, Z) voxel coordinates to center on.
    skeletons :     navis.TreeNeuron | navis.CatmaidNeuron | NeuronList
                    Skeleton(s) to add as annotation layer(s).
    seg_colors :    list | dict, optional
                    List or dictionary mapping colors to ``segments``.
    invis_segs :    int | list, optional
                    Selected but invisible segments.
    dataset :       'production' | 'sandbox'
                    Segmentation dataset to use.
    scene :         dict | str, optional
                    If you want to edit an existing scene, provide it either
                    as already decoded dictionary or as string that can be
                    interpreted by :func:`fafbseg.flywire.decode_url`.
    open_brower :   bool
                    If True, will open the url in a new tab of your webbrowser.
                    By default, we will first try to open in Google Chrome and
                    failing that fall back to your default browser.
    to_clipboard :  bool
                    If True, will copy URL to clipboard.
    short :         bool
                    If True, will make a shortened URL.

    Returns
    -------
    url :           str

    """
    # Translate "production"/"sandbox" into the corresponding dataset
    dataset = utils.FLYWIRE_DATASETS.get(dataset, dataset)

    # If scene provided as str, decode into dictionary
    if isinstance(scene, str):
        scene = decode_url(scene, ret='full')
    elif isinstance(scene, dict):
        # Do not modify original scene! We need to deepcopy here!
        scene = copy.deepcopy(scene)

    # If no scene provided, prepare the minimal scene
    if not scene:
        # Do not modify original scene! We need to deepcopy here!
        scene = copy.deepcopy(MINIMAL_SCENE)
        scene['layers'][1]['source'] = scene['layers'][1]['source'].format(dataset=dataset)

        if dataset == utils.FLYWIRE_DATASETS['sandbox']:
            scene['layers'][1]['name'] = 'sandbox-segmentation-FOR PRACTICE ONLY'
            scene['selectedLayer']['layer'] = 'sandbox-segmentation-FOR PRACTICE ONLY'

    # At this point scene HAS to be a dictionary
    if not isinstance(scene, dict):
        raise TypeError(f'Expected `scene` as dict or str, got "{type(scene)}"')

    # First add selected segments
    seg_layer_ix = [i for i, l in enumerate(scene['layers']) if l['type'] == 'segmentation_with_graph']
    if not seg_layer_ix:
        scene['layers'].append(MINIMAL_SCENE['layers'][1].copy())
        scene['layers'][-1]['source'] = scene['layers'][-1]['source'].format(dataset=dataset)
        seg_layer_ix = -1

        if dataset == utils.FLYWIRE_DATASETS['sandbox']:
            scene['layers'][-1]['name'] = 'sandbox-segmentation-FOR PRACTICE ONLY'
    else:
        seg_layer_ix = seg_layer_ix[0]

    # If provided, add segments
    if not isinstance(segments, type(None)):
        # Force to list and make strings
        segments = navis.utils.make_iterable(segments, force_type=str).tolist()

        # Add to, not replace already selected segments
        present = scene['layers'][seg_layer_ix].get('segments', [])
        scene['layers'][seg_layer_ix]['segments'] = present + segments

    if not isinstance(invis_segs, type(None)):
        # Force to list and make strings
        invis_segs = navis.utils.make_iterable(invis_segs, force_type=str).tolist()

        # Add to, not replace already selected segments
        present = scene['layers'][seg_layer_ix].get('hiddenSegments', [])
        scene['layers'][seg_layer_ix]['hiddenSegments'] = present + invis_segs

    # All present segments
    seg_layer = scene['layers'][seg_layer_ix]
    all_segs = seg_layer.get('segments', []) + seg_layer.get('hiddenSegments', [])

    # See if we need to assign colors
    if not isinstance(seg_colors, type(None)):
        if isinstance(seg_colors, str):
            seg_colors = {s: seg_colors for s in all_segs}
        elif isinstance(seg_colors, tuple) and len(seg_colors) == 3:
            seg_colors = {s: seg_colors for s in all_segs}
        elif not isinstance(seg_colors, dict):
            if not navis.utils.is_iterable(seg_colors):
                raise TypeError(f'`seg_colors` must be dict or iterable, got "{type(seg_colors)}"')
            if len(seg_colors) != len(all_segs):
                raise ValueError(f'Got {len(seg_colors)} colors for {len(all_segs)} segments.')

            # Turn into dictionary
            seg_colors = dict(zip(all_segs, seg_colors))

        # Turn colors into hex
        # Also make sure keys are int (not np.int64)
        seg_colors = {int(s): mcl.to_hex(c) for s, c in seg_colors.items()}

        # Assign colors
        scene['layers'][seg_layer_ix]['segmentColors'] = seg_colors

    # Set coordinates if provided
    if not isinstance(coords, type(None)):
        coords = np.asarray(coords)
        if not coords.ndim == 1 and coords.shape[0] == 3:
            raise ValueError('Expected coords to be an (3, ) array of x/y/z '
                             f'coordinates, got {coords.shape}')
        scene['navigation']['pose']['position']['voxelCoordinates'] = coords.round().astype(int).tolist()

    if not isinstance(annotations, type(None)):
        scene = add_annotation_layer(annotations, scene)

    if not isinstance(skeletons, type(None)):
        if isinstance(skeletons, navis.NeuronList):
            for n in skeletons:
                scene = add_skeleton_layer(n, scene)
        else:
            scene = add_skeleton_layer(skeletons, scene)

    if short:
        url = shorten_url(scene)
    else:
        scene_str = json.dumps(scene).replace("'",
                                              '"').replace("True",
                                                           "true").replace("False",
                                                                           "false")
        url = f'{NGL_URL}/#!{quote(scene_str)}'

    if open_browser:
        try:
            wb = webbrowser.get('chrome')
        except BaseException:
            wb = webbrowser

        wb.open_new_tab(url)

    if to_clipboard:
        pyperclip.copy(url)
        print('URL copied to clipboard.')

    return url


def add_skeleton_layer(x, scene):
    """Add skeleton as new layer to scene.

    Parameters
    ----------
    x :             navis.TreeNeuron | pymaid.CatmaidNeuron | int
                    Neuron to generate a URL for. Integers are interpreted as
                    CATMAID skeleton IDs. CatmaidNeurons will automatically be
                    transformed to FlyWire coordinates. Neurons are expected to
                    be in nanometers and will be converted to voxels.
    scene :         dict
                    Scene to add annotation layer to.

    Returns
    -------
    modified scene : dict

    """
    if not isinstance(scene, dict):
        raise TypeError(f'`scene` must be dict, got "{type(scene)}"')
    scene = scene.copy()

    if not isinstance(x, (navis.TreeNeuron, navis.NeuronList, pd.DataFrame)):
        x = pymaid.get_neuron(x)

    if isinstance(x, navis.NeuronList):
        if len(x) > 1:
            raise ValueError(f'Expected a single neuron, got {len(x)}')

    #if isinstance(x, pymaid.CatmaidNeuron):
    #    x = xform.fafb14_to_flywire(x, coordinates='nm')

    if not isinstance(x, (navis.TreeNeuron, pd.DataFrame)):
        raise TypeError(f'Expected skeleton, got {type(x)}')

    if isinstance(x, navis.TreeNeuron):
        nodes = x.nodes
    else:
        nodes = x

    # Generate list of segments
    not_root = nodes[nodes.parent_id >= 0]
    loc1 = not_root[['x', 'y', 'z']].values
    loc2 = nodes.set_index('node_id').loc[not_root.parent_id.values,
                                          ['x', 'y', 'z']].values
    stack = np.dstack((loc1, loc2))
    stack = np.transpose(stack, (0, 2, 1))

    stack = stack / [4, 4, 40]

    return add_annotation_layer(stack, scene)


def add_annotation_layer(annotations, scene, connected=False):
    """Add annotations as new layer to scene.

    Parameters
    ----------
    annotations :   numpy array
                    Coordinates [in 4x4x40 voxels] for annotations. The format
                    determines the type of annotation::
                        - point: (N, 3) of x/y/z coordinates
                        - line: (N, 2, 3) pairs x/y/z coordinates for start and
                          end point for each line segment
                        - ellipsoid: (N, 4) of x/y/z/radius

    scene :         dict
                    Scene to add annotation layer to.
    connected :     bool (TODO)
                    If True, point annotations will be treated as a segment of
                    connected points.

    Returns
    -------
    modified scene : dict

    """
    if not isinstance(scene, dict):
        raise TypeError(f'`scene` must be dict, got "{type(scene)}"')
    scene = scene.copy()

    annotations = np.asarray(annotations)

    # Generate records
    records = []
    if annotations.ndim == 2 and annotations.shape[1] == 3:
        for co in annotations.round().astype(int).tolist():
            records.append({'point': co,
                            'type': 'point',
                            'tagIds': [],
                            'id': str(uuid.uuid4())})
    elif annotations.ndim == 2 and annotations.shape[1] == 3:
        for co in annotations.round().astype(int).tolist():
            records.append({'center': co,
                            'type': 'ellipsoid',
                            'id': str(uuid.uuid4())})
    elif annotations.ndim == 3 and annotations.shape[1] == 2 and annotations.shape[2] == 3:
        for co in annotations.round().astype(int).tolist():
            start, end = co[0], co[1]
            records.append({'pointA': start,
                            'pointB': end,
                            'type': 'line',
                            'id': str(uuid.uuid4())})
    else:
        raise ValueError('Expected annotations to be x/y/z coordinates of either'
                         '(N, 3), (N, 4) or (N, 2, 3) shape for points, '
                         f'ellipsoids or lines, respectively. Got {annotations.shape}')

    existing_an_layers = [l for l in scene['layers'] if l['type'] == 'annotation']
    name = f'annotation{len(existing_an_layers)}'
    an_layer = {"type": "annotation",
                "annotations": records,
                "annotationTags": [],
                "voxelSize": [4, 4, 40],
                "name": name}

    scene['layers'].append(an_layer)

    return scene


def decode_url(url, ret='brief'):
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
    >>> flywire.decode_url('https://ngl.flywire.ai/?json_url=https://globalv1.flywire-daf.com/nglstate/6267328375291904')
    {'position': [132715.625, 55805.6796875, 3289.61181640625],
     'annotations': [],
     'selected': ['720575940621039145']}

    """
    assert isinstance(url, (str, dict))
    assert ret in ['brief', 'full']

    query = parse_qs(urlparse(url).query, keep_blank_values=True)

    if 'json_url' in query:
        # Fetch state
        token = utils.get_chunkedgraph_secret()
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


def shorten_url(scene, refresh_session=False):
    """Generate short url for given scene.

    Parameters
    ----------
    scene :             dict | str
                        Scene to encode as short URL. Can be dict or a full URL.
    refresh_session :   bool
                        If True will force refreshing the session.

    Returns
    -------
    shortened URL :  str

    """
    if not isinstance(scene, (dict, str)):
        raise TypeError(f'Expected `scene` to be dict or string, got "{type(scene)}"')

    if isinstance(scene, str):
        scene = decode_url(scene)

    global session

    if not session or refresh_session:
        session = requests.Session()
        # Load token
        token = utils.get_chunkedgraph_secret()

        # Generate header and cookie
        auth_header = {"Authorization": f"Bearer {token}"}
        session.headers.update(auth_header)
        cookie_obj = requests.cookies.create_cookie(name='middle_auth_token',
                                                    value=token)
        session.cookies.set_cookie(cookie_obj)

    # Upload state
    url = f'{STATE_URL}/post'
    resp = session.post(url, data=json.dumps(scene))
    resp.raise_for_status()

    return f'{NGL_URL}/?json_url={resp.json()}'


def generate_open_ends_url(x):
    """Generate a FlyWire URL with potential open ends for given neuron.

    Parameters
    ----------
    x :     root ID | navis.TreeNeuron | mesh
            ID of neuron to generate open ends for.

    """
    pass
