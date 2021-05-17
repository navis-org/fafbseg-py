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
import navis
import trimesh as tm

from .. import xform
from ..move import merge_into_catmaid

from .skeletonize import skeletonize_neuron
from .utils import parse_volume

try:
    import skeletor as sk
except ImportError:
    sk = None
except BaseException:
    raise

__all__ = ['merge_flywire_neuron']


def merge_flywire_neuron(id, target_instance, tag, flywire_dataset='production',
                         assert_id_match=True, drop_soma_hairball=True, **kwargs):
    """Merge flywire neuron into FAFB.

    This function (1) fetches a mesh from flywire, (2) turns it into a skeleton,
    (3) maps the coordinates to FAFB 14 and (4) runs ``fafbseg.merge_neuron``
    to merge the skeleton into CATMAID. See Examples below on how to run these
    individual steps yourself if you want more control over e.g. how the mesh
    is skeletonized.

    Parameters
    ----------
    id  :                int
                         ID of the flywire neuron you want to merge.
    target_instance :    pymaid.CatmaidInstance
                         Instance to merge the neuron into into.
    tag :                str
                         You personal tag to add as annotation once import into
                         CATMAID is complete.
    dataset :            str | CloudVolume
                         Against which flywire dataset to query::
                            - "production" (current production dataset, fly_v31)
                            - "sandbox" (i.e. fly_v26)
    assert_id_match :    bool
                         If True, will check if skeleton nodes map to the
                         correct segment ID and if not will move them back into
                         the segment. This is potentially very slow!
    drop_soma_hairball : bool
                         If True, we will try to drop the hairball that is
                         typically created inside the soma.
    **kwargs
                Keyword arguments are passed on to ``fafbseg.merge_neuron``.

    Examples
    --------
    # Import flywire neuron
    >>> _ = merge_flywire_neuron(id=720575940610453042,
    ...                          cvpath='graphene://https://prodv1.flywire-daf.com/segmentation/1.0/fly_v26',
    ...                          target_instance=manual,
    ...                          tag='WTCam')

    """
    if not sk:
        raise ImportError('Must install skeletor: pip3 install skeletor')

    vol = parse_volume(flywire_dataset)

    # Make sure this is a valid integer
    id = int(id)

    # Download the mesh
    mesh = vol.mesh.get(id, deduplicate_chunk_boundaries=False)[id]

    # Convert to neuron
    n_fw, simp, cntr = skeletonize_neuron(mesh,
                                          drop_soma_hairball=drop_soma_hairball,
                                          dataset=flywire_dataset,
                                          assert_id_match=assert_id_match)

    # Confirm
    viewer = navis.Viewer(title='Confirm skeletonization')
    # Make sure viewer is actually visible and cleared
    viewer.show()
    viewer.clear()
    # Add skeleton
    viewer.add(n_fw, color='r')

    msg = """
    Please carefully inspect the skeletonization of the flywire mesh.
    Hit ENTER to proceed if happy or CTRL-C to cancel.
    """

    # Add mesh last - otherwise it might mask out other objects despite alpha
    viewer.add(navis.MeshNeuron(mesh), color='w', alpha=.2)

    try:
        _ = input(msg)
    except KeyboardInterrupt:
        raise KeyboardInterrupt('Merge process aborted by user.')
    except BaseException:
        raise
    finally:
        viewer.close()

    # Xform to FAFB
    n_fafb = xform.flywire_to_fafb14(n_fw, on_fail='raise', coordinates='nm', inplace=False)
    mesh_fafb = xform.flywire_to_fafb14(tm.Trimesh(mesh.vertices, mesh.faces),
                                        on_fail='raise', coordinates='nm', inplace=False)

    # Heal neuron
    n_fafb = navis.heal_fragmented_neuron(n_fafb)

    # Merge neuron
    return merge_into_catmaid(n_fafb, target_instance=target_instance, tag=tag,
                              mesh=mesh_fafb, **kwargs)
