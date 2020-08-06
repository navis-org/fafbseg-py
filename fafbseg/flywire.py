# A collection of tools to interface with manually traced and autosegmented data
# in FAFB.
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

import cloudvolume as cv
import numpy as np

from urllib.parse import urlparse, parse_qs

from .mapping import xform_flywire_fafb14
from .merge import merge_neuron
from .segmentation import GSPointLoader

try:
    import skeletor as sk
except ImportError:
    sk = None
except BaseException:
    raise


def __decode_url(url):
    """Decode neuroglancer URL."""
    assert isinstance(url, str)

    query = parse_qs(urlparse(url).query)

    if 'json_url' in query:
        # Fetch state
        pass


def get_seg_ids(locs, root_ids=True, vol='graphene://https://prodv1.flywire-daf.com/segmentation/1.0/fly_v31',
                progress=True, coordinates='pixel', max_workers=8, **kwargs):
    """Retrieve flywire IDs at given location(s).

    Parameters
    ----------
    locs :          list-like
                    Array of x/y/z coordinates.
    root_ids :      bool
                    If True, will return root IDs. If False, will return supervoxel
                    IDs.
    vol :           str | CloudVolume
    progress :      bool
                    If True, shows progress bar.
    coordinates :   "pixel" | "nm"
                    Units in which your coordinates are in. "pixel" is assumed
                    to be 4x4x40 (x/y/z) nanometers.
    max_workers :   int
                    How many parallel requests we can make to the segmentation source.

    Returns
    -------
    list
                List of segmentation IDs in the same order as ``locs``.

    """
    assert coordinates in ['nm', 'pixel']

    locs = np.array(locs)
    assert locs.shape[1] == 3

    global fw_vol
    if 'CloudVolume' not in str(type(vol)):
        #  Change default volume if necessary
        if not fw_vol or getattr(fw_vol, 'path') != vol:
            # Set and update defaults from kwargs
            defaults = dict(cache=True,
                            mip=0,
                            fill_missing=True,
                            progress=False)
            defaults.update(kwargs)

            fw_vol = cv.CloudVolume(vol, **defaults)
            fw_vol.path = vol
    else:
        fw_vol = vol

    # GSPointLoader expects nanometer
    if coordinates == 'pixel':
        #res = vol.info['scales'][vol.mip]['resolution']
        locs = (locs * [4, 4, 40]).astype(int)

    pl = GSPointLoader(fw_vol)
    pl.add_points(locs)

    svoxels, data = pl.load_all(max_workers=max_workers,
                                progress=progress,
                                return_sorted=True)

    if root_ids:
        return fw_vol.get_roots(svoxels)

    return data


def __merge_flywire_neuron(id, cvpath, **kwargs):
    """Merge flywire neuron into FAFB.

    This function (1) fetches a mesh from flywire, (2) turns it into a skeleton,
    (3) maps the coordinates to FAFB 14 and (4) runs ``fafbseg.merge_neuron``
    to merge the skeleton into CATMAID. See Examples below on how to run these
    individual steps yourself if you want more control over e.g. how the mesh
    is skeletonized.

    Parameters
    ----------
    id  :       int
                ID of the neuron you want to merge.
    cvpath :    str | cloudvolume.CloudVolume
                Either the path to the flywire segmentation (``graphene://...``)
                or an already initialized ``CloudVolume``.
    **kwargs
                Keyword arguments are passed on to ``fafbseg.merge_neuron``.

    Examples
    --------
    # Import flywire neuron
    >>> _ = merge_flywire_neuron(id=720575940610453042,
    ...                          cvpath='graphene://https://prodv1.flywire-daf.com/segmentation/1.0/fly_v26',
    ...                          target_instance=manual,
    ...                          tag='WTCam')

    # Run each step yourself
    >>>

    """
    if not sk:
        raise ImportError('Must install skeletor: pip3 install skeletor')

    if isinstance(cvpath, cv.CloudVolume):
        vol = cvpath
    elif isinstance(cvpath, str):
        vol = cv.CloudVolume(cvpath)
    else:
        raise TypeError('Unable to initialize a cloudvolume from "{}"'.format(type(cvpath)))

    # Make sure this is a valid integer
    id = int(id)

    # Download the mesh
    mesh = vol.mesh.get(id)[id]

    # Contract
    cntr = sk.contract(mesh)

    # Generate skeleton
    swc = sk.skeletonize(cntr, method='vertex_clustering', sampling_dist=100)

    # Clean up
    cleaned = sk.clean(swc, mesh=mesh)

    # Extract radii
    cleaned['radius'] = sk.radius(cleaned, mesh=mesh)

    # Convert to neuron
    n_fw = pymaid.from_swc(cleaned, neuron_id=id)

    # Xform to FAFB
    n_fafb = xform_flywire_fafb14(n_fw, on_fail='raise', coordinates='nm', inplace=False)

    # Merge neuron
    return merge_neuron(n_fafb, **kwargs)


fw_vol = None
