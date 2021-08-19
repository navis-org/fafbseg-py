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

"""Functions to fetch and set annotations in FlyWire using the annotation
framework and the materialization engine."""

import navis

import numpy as np
import pandas as pd

from .utils import get_cave_client


__all__ = ['get_somas', 'get_materialization_versions']


def get_materialization_versions(dataset='production'):
    """Fetch info on the available materializations."""
    # Get/Initialize the CAVE client
    client = get_cave_client(dataset)

    # Get currently existing versions
    versions = client.materialize.get_versions()

    # Fetch meta data
    meta = pd.DataFrame.from_records([client.materialize.get_version_metadata(v) for v in versions])

    # Reduce precision for timestamps to make more readable
    meta['expires_on'] = meta.expires_on.values.astype('datetime64[m]')
    meta['time_stamp'] = meta.time_stamp.values.astype('datetime64[m]')

    return meta.sort_values('version', ascending=False).reset_index(drop=True)


def get_somas(root_ids, mat_id=None, dataset='production'):
    """Fetch nuclei segmentation for given neuron(s).

    A couple notes:
      1. This uses the materialization engine to search for root IDs. This
         engine is always slightly lagging behind the live data. If you need
         to-the-minute info your best bet is to fetch the entire table (with
         `root_ids=None`) and update the roots based on the supervoxel
         associated with each nucleus (`flywire.supervoxels_to_roots`).
      2. Since this is a nucleus detection you will find that some neurons do
         not have an entry despite having a soma. This is due to the
         "avocado problem" where the nucleus is separate from the rest of the
         soma.

    Parameters
    ----------
    root_ids  :         int | list of ints | None
                        FlyWire root ID(s) for which to fetch soma infos. Use
                        ``None`` to fetch complete list of annotated nuclei.

    Returns
    -------
    DataFrame
                        Pandas DataFrame with nucleu (see Examples). Root IDs
                        without a nucleus will be missing. The `rad_est` is
                        the estimated radius based on the wides point of the
                        nucleus. Take it with a grain of salt!

    Examples
    --------
    >>> from fafbseg import flywire
    >>> info = flywire.get_somas([720575940614131061])
    >>> info
            id valid   pt_supervoxel_id          pt_root_id     volume              pt_position  rad_est
    0  7393349     t  82827379285852979  720575940630194247  26.141245  [709888, 227744, 57160]   2260.0
    1  7415013     t  83038760463398837  720575940632921242  53.711176  [722912, 244032, 65200]   2640.0

    """
    # Get/Initialize the CAVE client
    client = get_cave_client(dataset)

    filter_in_dict = None
    if not isinstance(root_ids, type(None)):
        root_ids = navis.utils.make_iterable(root_ids)
        filter_in_dict = {'pt_root_id': root_ids}

    nuc = client.materialize.query_table('nuclei_v1',
                                         filter_in_dict=filter_in_dict)

    # Add estimated radius based on nucleus
    if not nuc.empty:
        start = np.vstack(nuc.bb_start_position)
        end = np.vstack(nuc.bb_end_position)
        nuc['rad_est'] = np.abs(start - end).max(axis=1) / 2

    return nuc.drop(['bb_start_position', 'bb_end_position'], axis=1)
