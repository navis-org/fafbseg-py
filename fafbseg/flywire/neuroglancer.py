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
import seaborn as sns

from pathlib import Path
from functools import lru_cache
from urllib.parse import urlparse, parse_qs, quote, unquote

from . import utils
from .segmentation import neuron_to_segments
from .annotations import parse_neuroncriteria
from ..utils import make_iterable

__all__ = ["decode_url", "encode_url"]

# Data stuff
fp = Path(__file__).parent
data_path = fp.parent / "data"
session = None


@parse_neuroncriteria()
@utils.inject_dataset()
def encode_url(
    segments=None,
    annotations=None,
    coords=None,
    skeletons=None,
    seg_colors=None,
    seg_groups=None,
    invis_segs=None,
    scene=None,
    base_neuroglancer=False,
    layout='3d',
    open=False,
    to_clipboard=False,
    shorten=True,
    *,
    dataset=None,
):
    """Encode data as FlyWire neuroglancer scene.

    Parameters
    ----------
    segments :      int | list of int | NeuronCriteria, optional
                    Segment IDs to have selected.
    annotations :   (N, 3) array or dict of {name: (N, 3) array}, optional
                    Array or dict of xyz coordinates that will be added as
                    annotation layer(s). Should be in voxel coordinates. If you
                    need more control over this, see :func:`fafbseg.flywire.add_annotation_layer`.
    coords :        (3, ) array, optional
                    (X, Y, Z) voxel coordinates to center on.
    skeletons :     navis.TreeNeuron | navis.CatmaidNeuron | NeuronList
                    Skeleton(s) to add as annotation layer(s).
    seg_colors :    str | tuple | list | dict, optional
                    Single color (name or RGB tuple), or list or dictionary
                    mapping colors to ``segments``. Can also be a numpy array
                    of labels which will be automatically turned into colors.
    seg_groups :    list | dict, optional
                    List or dictionary mapping segments to groups. Each group
                    will get its own annotation layer.
    invis_segs :    int | list, optional
                    Selected but invisible segments.
    scene :         dict | str, optional
                    If you want to edit an existing scene, provide it either
                    as already decoded dictionary or as string that can be
                    interpreted by :func:`fafbseg.flywire.decode_url`.
    open :          bool
                    If True, will open the url in a new tab of your webbrowser.
                    By default, we will first try to open in Google Chrome and
                    failing that fall back to your default browser.
    to_clipboard :  bool
                    If True, will copy URL to clipboard.
    shorten :       bool
                    If True, will make a shortened URL.
    base_neuroglancer :  bool
                    Whether to use the base neuroglancer instead of the modified
                    FlyWire neuroglancer.
    layout :        "3d" | "xy-3d" | "xy"
                    Layout to show.
    dataset :       "public" | "production" | "sandbox" | "flat_630", optional
                    Against which FlyWire dataset to query. If ``None`` will fall
                    back to the default dataset (see
                    :func:`~fafbseg.flywire.set_default_dataset`).

    Returns
    -------
    url :           str

    """
    # If scene provided as str, decode into dictionary
    if isinstance(scene, str):
        scene = decode_url(scene, ret="full")

        # Find the segmentation layer (will raise if not found)
        seg_layer_ix = _find_flywire_layer(scene["layers"])
    elif isinstance(scene, dict):
        # Do not modify original scene! We need to deepcopy here!
        scene = copy.deepcopy(scene)

        # Find the segmentation layer (will raise if not found)
        seg_layer_ix = _find_flywire_layer(scene["layers"])
    elif scene is None:
        # Get the canned scene
        scene = construct_scene(
            dataset=dataset,
            segmentation=True,
            image=True,
            brain_mesh=True,
            layout=layout,
            base_neuroglancer=base_neuroglancer,
        )

        # Since we constructed this scene ourselves, we know which is the
        # segmentation layer
        seg_layer_ix = 1
    else:
        raise TypeError(f"`scene` must be string, dict or None, got {type(scene)}")

    # Set layout
    scene['layout'] = layout

    # Now we can start adding stuff to our scene
    # First, add segments (if applicable)
    if not isinstance(segments, type(None)):
        # Force to list and make strings
        segments = make_iterable(segments, force_type=str).tolist()

        # Add to, not replace already selected segments!
        if isinstance(seg_groups, type(None)):
            present = scene["layers"][seg_layer_ix].get("segments", [])
            scene["layers"][seg_layer_ix]["segments"] = present + segments

    # If we have `seg_groups` we will need to re-arrange segments into
    # individual layers
    if seg_groups is not None:
        # If seg_groups isn't already a dictionary, we need to turn it into one
        if not isinstance(seg_groups, dict):
            # Complain if this is not a container of appropriate length
            if not navis.utils.is_iterable(seg_groups):
                raise TypeError(
                    f'`seg_groups` must be dict or iterable, got "{type(seg_groups)}"'
                )
            if len(seg_groups) != len(segments):
                raise ValueError(
                    f"Got {len(seg_groups)} groups for {len(segments)} segments."
                )

            # Turn into array
            seg_groups = np.asarray(seg_groups)

            # If datatype is not object (i.e. not string), we will turn it into that
            if seg_groups.dtype != object:
                seg_groups = [f"group_{i}" for i in seg_groups]

            # Turn into dictionary
            seg_groups = dict(zip(segments, seg_groups))

            if len(seg_groups) != len(segments):
                print('Some segments seem to belong to multiple groups. This is '
                      'currently not supported.')

        # Check if dict is {id: group} or {group: [id1, id2, id3]} and force
        # into the latter
        is_list = [
            isinstance(v, (list, tuple, set, np.ndarray)) for v in seg_groups.values()
        ]
        if not any(is_list):
            groups = {}
            for s, g in seg_groups.items():
                if not isinstance(g, str):
                    raise TypeError(
                        f"Expected `seg_groups` to be strings, got {type(g)}"
                    )
                groups[g] = groups.get(g, []) + [s]
        elif all(is_list):
            groups = seg_groups
        else:
            raise ValueError(
                "`seg_groups` appears to be a mix of {id: group} "
                "and {group: [id1, id2, id3]}."
            )

        # Make a copy of the first segmentation layer and add the segments
        for g in groups:
            scene["layers"].append(copy.deepcopy(scene["layers"][seg_layer_ix]))
            scene["layers"][-1]["name"] = f"{g}"
            scene["layers"][-1]["segments"] = [str(s) for s in groups[g]]
            scene["layers"][-1]["visible"] = False

    # Add invisible segments
    if not isinstance(invis_segs, type(None)):
        # Force to list and make strings
        invis_segs = make_iterable(invis_segs, force_type=str).tolist()

        # Add to, not replace already selected segments
        present = scene["layers"][seg_layer_ix].get("hiddenSegments", [])
        scene["layers"][seg_layer_ix]["hiddenSegments"] = present + invis_segs

    # All present and visible segments
    all_segs = segments

    # See if we need to assign colors
    if seg_colors is not None:
        # Parse color(s)
        # 1. Single color (e.g. "white")
        if isinstance(seg_colors, str):
            seg_colors = {s: seg_colors for s in all_segs}
        # 2. A single (r, g, b) color
        elif isinstance(seg_colors, tuple) and len(seg_colors) == 3:
            seg_colors = {s: seg_colors for s in all_segs}
        # 3. A (N, ) list of labels
        elif (
            isinstance(seg_colors, (np.ndarray, pd.Series, pd.Categorical))
            and seg_colors.ndim == 1
        ):
            if len(seg_colors) != len(all_segs):
                raise ValueError(
                    f"Got {len(seg_colors)} colors for {len(all_segs)} segments."
                )

            # Number of unique labels - so we can find a good palette
            uni_ = np.unique(seg_colors)
            if len(uni_) > 20:
                # Note the +1 to avoid starting and ending on the same color
                pal = sns.color_palette("hls", len(uni_) + 1)
                # Shuffle to avoid having two neighbouring clusters with
                # similar colours
                rng = np.random.default_rng(1985)
                rng.shuffle(pal)
            elif len(uni_) > 10:
                pal = sns.color_palette("tab20", len(uni_))
            else:
                pal = sns.color_palette("tab10", len(uni_))
            _colors = dict(zip(uni_, pal))
            seg_colors = {s: _colors[l] for s, l in zip(all_segs, seg_colors)}
        # 4. Anything else that isn't already a dictionary
        elif not isinstance(seg_colors, dict):
            if not navis.utils.is_iterable(seg_colors):
                raise TypeError(
                    f'`seg_colors` must be dict or iterable, got "{type(seg_colors)}"'
                )
            if len(seg_colors) < len(all_segs):
                raise ValueError(
                    f"Got {len(seg_colors)} colors for {len(all_segs)} segments."
                )

            # Turn into dictionary
            seg_colors = dict(zip(all_segs, seg_colors))

        # Turn colors into hex codes
        # Also make sure keys are int (not np.int64)
        # Not sure but this might cause issue on Windows systems
        # But JSON doesn't like np.int64... so we're screwed either way
        seg_colors = {str(s): mcl.to_hex(c) for s, c in seg_colors.items()}

        # Assign colors
        scene["layers"][seg_layer_ix]["segmentColors"] = seg_colors

        # Also propagate colors to seg_groups (if applicable)
        if seg_groups is not None:
            for l in scene["layers"]:
                if l["name"] in groups:
                    l["segmentColors"] = {s: seg_colors[s] for s in l["segments"]}

    # Set coordinates if provided
    if coords is not None:
        coords = np.asarray(coords)
        if not coords.ndim == 1 and coords.shape[0] == 3:
            raise ValueError(
                "Expected coords to be an (3, ) array of x/y/z "
                f"coordinates, got {coords.shape}"
            )
        scene["navigation"]["pose"]["position"]["voxelCoordinates"] = (
            coords.round().astype(int).tolist()
        )

    # Add annotations if provided
    if annotations is not None:
        if isinstance(annotations, (np.ndarray, list)):
            scene = add_annotation_layer(annotations, scene)
        elif isinstance(annotations, dict):
            for layer, an in annotations.items():
                scene = add_annotation_layer(an, scene, name=layer)

    # Add skeletons if provided
    if skeletons is not None:
        if isinstance(skeletons, navis.NeuronList):
            for n in skeletons:
                scene = add_skeleton_layer(n, scene)
        else:
            scene = add_skeleton_layer(skeletons, scene)

    return scene_to_url(scene, base_neuroglancer=base_neuroglancer,
                        shorten=shorten, open=open, to_clipboard=to_clipboard)


def scene_to_url(scene, base_neuroglancer=False, shorten=True, open=False, to_clipboard=False):
    """Turn neuroglancer scene into a URL.

    Parameter
    ---------
    scene :             dict
                        Scene to convert.
    base_neuroglancer : bool
                        Whether to use base over the modified FlyWire neuroglancer.
    shorten :           bool
                        Whether to shorten the URL. Currently only works with
                        FlyWire neuroglancer.
    open :              bool
                        Whether to open the URL in the browser.
    to_clipboard :      bool
                        Whether to also copy the URL to the clipboard.

    Returns
    -------
    url :               str

    """
    NGL_SCENES = copy.deepcopy(_load_ngl_scenes())

    # We currently have no official base neuroglancer link shortener
    if base_neuroglancer and shorten:
        print(
            "It is currently only possible to shorten links for the FlyWire "
            "neuroglancer"
        )
        shorten = False

    # Shorten URL
    if shorten:
        url = shorten_url(scene)
    else:
        scene_str = (
            json.dumps(scene)
            .replace("'", '"')
            .replace("True", "true")
            .replace("False", "false")
        )

        # Turn e.g. "flywire" ngl_url into ""https://ngl.flywire.ai""
        if not base_neuroglancer:
            ngl_url = NGL_SCENES["NGL_URL_FLYWIRE"]
        else:
            ngl_url = NGL_SCENES["NGL_URL_BASIC"]

        url = f"{ngl_url}/#!{quote(scene_str)}"

    # Open in web browser
    if open:
        try:
            wb = webbrowser.get("chrome")
        except BaseException:
            wb = webbrowser

        wb.open_new_tab(url)

    # Copy to Clipboard
    if to_clipboard:
        pyperclip.copy(url)
        print("URL copied to clipboard.")

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
            raise ValueError(f"Expected a single neuron, got {len(x)}")

    # if isinstance(x, pymaid.CatmaidNeuron):
    #    x = xform.fafb14_to_flywire(x, coordinates='nm')

    if not isinstance(x, (navis.TreeNeuron, pd.DataFrame)):
        raise TypeError(f"Expected skeleton, got {type(x)}")

    if isinstance(x, navis.TreeNeuron):
        nodes = x.nodes
    else:
        nodes = x

    # Generate list of segments
    not_root = nodes[nodes.parent_id >= 0]
    loc1 = not_root[["x", "y", "z"]].values
    loc2 = (
        nodes.set_index("node_id")
        .loc[not_root.parent_id.values, ["x", "y", "z"]]
        .values
    )
    stack = np.dstack((loc1, loc2))
    stack = np.transpose(stack, (0, 2, 1))

    stack = stack / [4, 4, 40]

    return add_annotation_layer(stack, scene)


def add_annotation_layer(annotations, scene, name=None, connected=False):
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
    name :          str
                    Name of the annotation layer.
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
            records.append(
                {"point": co, "type": "point", "tagIds": [], "id": str(uuid.uuid4())}
            )
    elif annotations.ndim == 2 and annotations.shape[1] == 3:
        for co in annotations.round().astype(int).tolist():
            records.append({"center": co, "type": "ellipsoid", "id": str(uuid.uuid4())})
    elif (
        annotations.ndim == 3
        and annotations.shape[1] == 2
        and annotations.shape[2] == 3
    ):
        for co in annotations.round().astype(int).tolist():
            start, end = co[0], co[1]
            records.append(
                {
                    "pointA": start,
                    "pointB": end,
                    "type": "line",
                    "id": str(uuid.uuid4()),
                }
            )
    else:
        raise ValueError(
            "Expected annotations to be x/y/z coordinates of either"
            "(N, 3), (N, 4) or (N, 2, 3) shape for points, "
            f"ellipsoids or lines, respectively. Got {annotations.shape}"
        )

    if not name:
        existing_an_layers = [l for l in scene["layers"] if l["type"] == "annotation"]
        name = f"annotation{len(existing_an_layers)}"

    an_layer = {
        "type": "annotation",
        "annotations": records,
        "annotationTags": [],
        "voxelSize": [4, 4, 40],
        "name": name,
    }

    scene["layers"].append(an_layer)

    return scene


def decode_url(url, format="json"):
    """Decode neuroglancer URL.

    Parameters
    ----------
    url :       str | list of str
                URL(s) to decode. Can be shortened URL. Note that not all
                `formats` work with multiple URLs.
    format :    "json" | "brief" | "dataframe"
                What to return:
                 - "json" (default) returns the full JSON
                 - "brief" only returns position (in voxels), selected segments
                   and annotations
                 - "dataframe" returns a frame with segment IDs and which
                   layers they came from

    Returns
    -------
    dict
                If format is "json" or "brief".
    DataFrame
                If format='dataframe'.

    Examples
    --------
    >>> from fafbseg import flywire
    >>> flywire.decode_url('https://ngl.flywire.ai/?json_url=https://globalv1.flywire-daf.com/nglstate/6267328375291904',
    ...                    format='brief')
    {'position': [132715.625, 55805.6796875, 3289.61181640625],
     'annotations': [],
     'selected': ['720575940621039145']}

    """
    if isinstance(url, list):
        if format != "dataframe":
            raise ValueError('Can only parse multiple URLs if format="dataframe"')
        return pd.concat([decode_url(u, format=format) for u in url], axis=0)

    if not isinstance(url, (str, dict)):
        raise TypeError(f'`url` must be string, got "{type(url)}"')

    # Parse FlyWire URL
    if "json_url" in url:
        # Fetch state
        token = utils.get_chunkedgraph_secret()
        query = parse_qs(urlparse(url).query, keep_blank_values=True)
        r = requests.get(
            query["json_url"][0], headers={"Authorization": f"Bearer {token}"}
        )
        r.raise_for_status()

        scene = r.json()
    # Parse URLs with a link to Google buckets
    elif '!gs://' in url:
        path = urlparse(url).fragment.replace('!gs://', '')
        r = requests.get(f'https://storage.googleapis.com/{path}')
        r.raise_for_status()

        scene = r.json()
    elif isinstance(url, str):
        query = unquote(urlparse(url).fragment)[1:]
        scene = json.loads(query)
    else:
        scene = url

    # "full" is for legacy purposes
    if format in ('json', 'full'):
        return scene
    elif format == "brief":
        seg_layers = [
            layer
            for layer in scene.get("layers", [])
            if "segmentation" in layer.get("type")
        ]
        an_layers = [
            layer
            for layer in scene.get("layers", [])
            if layer.get("type") == "annotation"
        ]
        try:
            position = scene["navigation"]["pose"]["position"].get(
                "voxelCoordinates", None
            )
        except KeyError:
            position = None

        return {
            "position": position,
            "annotations": [
                a for layer in an_layers for a in layer.get("annotations", [])
            ],
            "selected": [s for layer in seg_layers for s in layer.get("segments", [])],
        }
    elif format == "dataframe":
        segs = []
        seg_layers = [
            layer for layer in scene["layers"] if "segmentation" in layer.get("type")
        ]
        for layer in seg_layers:
            for s in layer.get("segments", []):
                segs.append([int(s.replace('!', '')), layer["name"], not s.startswith('!')])

        return pd.DataFrame(segs, columns=["segment", "layer", "visible"])
    else:
        raise ValueError(f'Unexpected format: "{format}')

    return scene


def shorten_url(scene, state_url=None, refresh_session=False):
    """Generate short url for given scene.

    Parameters
    ----------
    scene :             dict | str
                        Scene to encode as short URL. Can be dict or a full URL.
    state_url :         str, optional
                        URL for the state server. If not provided will use the
                        default state server for FlyWire.
    refresh_session :   bool
                        If True will force refreshing the session.

    Returns
    -------
    shortened URL :  str

    """
    # Make a deepcopy of everything - avoids having to do this later
    NGL_SCENES = copy.deepcopy(_load_ngl_scenes())

    if not isinstance(scene, (dict, str)):
        raise TypeError(f'Expected `scene` to be dict or string, got "{type(scene)}"')

    if not state_url:
        state_url = NGL_SCENES["FLYWIRE_STATE_URL"]

    ngl_url = NGL_SCENES["NGL_URL_FLYWIRE"]

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
        cookie_obj = requests.cookies.create_cookie(
            name="middle_auth_token", value=token
        )
        session.cookies.set_cookie(cookie_obj)

    # Upload state
    url = f"{state_url}/post"
    resp = session.post(url, data=json.dumps(scene))
    resp.raise_for_status()

    return f"{ngl_url}/?json_url={resp.json()}"


def neurons_to_url(x, top_N=1, downsample=False, coordinates="nm"):
    """Find FlyWire segments overlapping with given neuron(s) and create URLs.

    Parameters
    ----------
    x :             NeuronList w/ TreeNeurons
                    Must be in FlyWire (FAFB14.1) nanometer space.
    top_N :         int, float
                    How many overlapping fragments to include in the URL. If >= 1
                    will treat it as the top N fragments. If < 1 will treat as "all
                    fragments that collectively make up this fraction of the neuron".
    downsample :    int, optional
                    Factor by which to downsample the skeleton before adding to
                    FlyWire scene.

    Returns
    -------
    pandas.DataFrame

    """
    assert isinstance(x, navis.NeuronList)
    assert not x.is_degenerated
    assert isinstance(x[0], navis.TreeNeuron)

    ol = neuron_to_segments(x, coordinates=coordinates)

    data = []
    for n in navis.config.tqdm(x, desc="Creating URLs"):
        if n.id not in ol.columns:
            print(
                f"No overlapping fragments found for neuron {n.label}. Check "
                "`coordinates` parameter?"
            )

        this = ol[n.id].sort_values(ascending=False)
        pct = this / this.sum()

        if top_N >= 1:
            to_add = this.index[:top_N]
        else:
            to_add = this.index[: np.where(pct.cumsum() > top_N)[0][0] + 1]

        if downsample:
            n = navis.downsample_neuron(n, downsample)

        url = encode_url(segments=to_add, skeletons=n)

        row = [n.id, n.name, url]

        if top_N >= 1:
            for i in to_add:
                row += [i, pct.loc[i]]
        else:
            row.append(len(to_add))

        data.append(row)

    cols = ["id", "name", "url"]
    if top_N >= 1:
        for i in range(top_N):
            cols += [f"seg_{i + 1}", f"conf_{i + 1}"]
    else:
        cols.append("n_segs")

    return pd.DataFrame(data, columns=cols)


@lru_cache
def _load_ngl_scenes():
    """Load neuroglancer layers & settings."""
    with open(data_path / "ngl_scenes.json") as f:
        return json.load(f)


def _find_flywire_layer(layers, raise_not_found=True):
    """Find the FlyWire segmentation layer among given layers."""
    poss_names = list(utils.FLYWIRE_DATASETS.values())
    for i, layer in enumerate(layers):
        if layer["type"] == "segmentation_with_graph":
            return i
        if layer["type"] == "segmentation":
            if (layer["name"] in poss_names or 'flywire' in layer['name']):
                return i
    if raise_not_found:
        raise ValueError("Unable to identify flywire segmentation among layers")


def construct_scene(
    *,
    image=True,
    segmentation=True,
    hemibrain=False,
    hemibrain_mirror=False,
    brain_mesh=True,
    neuropils=False,
    hemibrain_neuropils=False,
    base_neuroglancer=False,
    layout='xy-3d',
    dataset="production",
):
    """Construct a basic neuroglancer scene.

    Parameters
    ----------
    image :         bool
                    Whether to add a layer for EM image data.
    segmentation :  bool
                    Whether to add a layer for the FlyWire segmentation.
    hemibrain :     bool
                    Whether to add a layer for hemibrain neuron meshes.
    hemibrain_mirror : bool
                    Whether to add a layer for mirrored hemibrain neuron meshes.
    brain_mesh :    bool
                    Whether to add a layer brain mesh.
    neuropils :     bool
                    Whether to add layer for FlyWire neuropils.
    hemibrain_neuropils : bool
                    Whether to add layer with hemibrain neuropils (including
                    hemibrain outline).
    base_neuroglancer : bool
                    Whether scene is used for the base neuroglancer as this
                    requires changes to the source for the segmentation.
    layout :        "3d" | "xy-3d" | "xy"
                    Layout to show.
    dataset :       "public" | "production" | "sandbox" | "flat_630" | "flat_783"
                    Which segmentation dataset to use.

    Returns
    -------
    scene :         dict
                    Layers in order of appearance:
                      1. Image layer
                      2. Segmentation layer
                      3. Hemibrain mesh layer
                      4. Mirrored hemibrain mesh layer
                      5. Brain mesh

    """
    # Make a deepcopy of everything - avoids having to do this later
    NGL_SCENES = copy.deepcopy(_load_ngl_scenes())

    # Get the canned scene
    scene = NGL_SCENES["MINIMAL_SCENE"]

    scene['layout'] = layout

    # Add image layer
    if image:
        scene["layers"].append(NGL_SCENES["FLYWIRE_IMAGE_LAYER"])

    # Add segmentation layer
    if segmentation:
        if dataset == "flat_630":
            scene["layers"].append(NGL_SCENES["FLYWIRE_SEG_LAYER_FLAT_630"])
        elif dataset == "flat_783":
            scene["layers"].append(NGL_SCENES["FLYWIRE_SEG_LAYER_FLAT_783"])
        else:
            if not base_neuroglancer:
                scene["layers"].append(NGL_SCENES["FLYWIRE_SEG_LAYER"])
            else:
                scene["layers"].append(NGL_SCENES["FLYWIRE_SEG_LAYER_BASIC_NGL"])

            # Set dataset (i.e. "sandbox" -> "fly_v26")
            if isinstance(scene["layers"][-1]["source"], str):
                scene["layers"][-1]["source"] = scene["layers"][-1]["source"].format(
                    dataset=utils.FLYWIRE_DATASETS.get(dataset, dataset)
                )
            elif isinstance(scene["layers"][-1]["source"], dict):
                scene["layers"][-1]["source"]["url"] = scene["layers"][-1]["source"][
                    "url"
                ].format(dataset=utils.FLYWIRE_DATASETS.get(dataset, dataset))
            elif isinstance(scene["layers"][-1]["source"], list):
                scene["layers"][-1]["source"][0]["url"] = scene["layers"][-1]["source"][
                    0
                ]["url"].format(dataset=utils.FLYWIRE_DATASETS.get(dataset, dataset))
            else:
                raise TypeError(
                    "Unexpected format for segmentation layer source: "
                    f"{type(scene['layers'][-1]['source'])}"
                )

        # Set segment layer name
        scene["layers"][-1]["name"] = utils.FLYWIRE_DATASETS.get(dataset, dataset)

    # Add hemibrain neuron mesh layer
    if hemibrain:
        scene["layers"].append(NGL_SCENES["HEMIBRAIN_MESH_LAYER"])

    # Add mirrored hemibrain neuron mesh layer
    if hemibrain_mirror:
        scene["layers"].append(NGL_SCENES["HEMIBRAIN_MESH_LAYER_MIRRORED"])

    # Add brain outline layer
    if brain_mesh:
        scene["layers"].append(NGL_SCENES["FLYWIRE_BRAIN_LAYER"])

    # Add FlyWire neuropils
    if neuropils:
        scene["layers"].append(NGL_SCENES["FLYWIRE_NEUROPILS_LAYER"])

    # Add hemibrain neuropils layer
    if hemibrain_neuropils:
        scene["layers"].append(NGL_SCENES["HEMIBRAIN_NEUROPILS_LAYER"])

    return scene
