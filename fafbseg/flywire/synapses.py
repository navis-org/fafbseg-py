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

import numpy as np

from .segmentation import roots_to_supervoxels, supervoxels_to_roots

from ..synapses.utils import catmaid_table
from .. import spine

__all__ = ['fetch_synapses', 'fetch_connectivity']


def fetch_synapses(x, attach=True, dataset='production', progress=True):
    """Fetch  Buhmann et al. (2019) synapses for given neuron(s).

    Uses a service on spine.janelia.org hosted by Eric Perlman and Davi Bock.

    Parameters
    ----------
    x :         int | list of int | Neuron/List
                Either a flywire segment ID (i.e. root ID), a list thereof or
                a Neuron/List. For neurons, the ``.id`` is assumed to be the
                root ID.
    attach :    bool
                If True and ``x`` is Neuron/List, the synapses will be added
                as ``.connectors`` table. For TreeNeurons (skeletons), the
                synapses will be mapped to the closest node.
    dataset :   str | CloudVolume
                Against which flywire dataset to query::
                    - "production" (current production dataset, fly_v31)
                    - "sandbox" (i.e. fly_v26)

    """
    # TODO
    pass


def fetch_connectivity(x, dataset='production', clean=True, style='catmaid',
                       min_score=30, progress=True):
    """Fetch  Buhmann et al. (2019) connectivity for given neuron(s).

    Uses a service on spine.janelia.org hosted by Eric Perlman and Davi Bock.

    Parameters
    ----------
    x :         int | list of int | Neuron/List
                Either a flywire segment ID (i.e. root ID), a list thereof or
                a Neuron/List. For neurons, the ``.id`` is assumed to be the
                root ID.
    dataset :   str | CloudVolume
                Against which flywire dataset to query::
                    - "production" (current production dataset, fly_v31)
                    - "sandbox" (i.e. fly_v26)
    clean :     bool
                If True, we will perform some clean up of the connectivity
                table:

    style :     "simple" | "catmaid"
                Style of the returned table.

    Returns
    -------
    pd.DataFrame
                Connectivity table.

    """
    if isinstance(x, navis.BaseNeuron):
        ids = [x.id]
    elif isinstance(x, navis.NeuronList):
        ids = x.id
    elif isinstance(x, (int, np.int)):
        ids = [x]
    else:
        ids = navis.utils.make_iterable(x)

    # Make sure we are working with proper numerical IDs
    try:
        ids = np.asarray(ids).astype(int)
    except ValueError:
        raise ValueError(f'Unable to convert IDs to integer: {ids}')
    except BaseException:
        raise

    # Now get supervoxels for these root IDs
    # (this is a dict)
    svoxels = roots_to_supervoxels(ids, dataset=dataset, progress=progress)

    # Query the synapses
    syn = spine.synapses.get_connectivity(svoxels, segmentation='flywire_supervoxels')

    # Next we need to run some clean-up:
    # 1. Drop below threshold connections
    if min_score:
        syn = syn[syn.cleft_scores >= min_score]
    # 2. Drop connections involving 0 (background, glia)
    syn = syn[(syn.pre != 0) & (syn.post != 0)]

    # Avoid copy warning
    syn = syn.copy()

    # Now map the supervoxels to root IDs
    svoxels = np.unique(syn[['pre', 'post']].values.flatten())
    roots = supervoxels_to_roots(svoxels, dataset=dataset)

    dct = dict(zip(svoxels, roots))

    syn['pre'] = syn.pre.map(dct)
    syn['post'] = syn.post.map(dct)

    # Turn into connectivity table
    cn_table = syn.groupby(['pre', 'post'], as_index=False).size().to_frame().reset_index(drop=False).rename({0: 'weight'}, axis=1)

    # Style
    if style == 'catmaid':
        cn_table = catmaid_table(cn_table, ids)
    else:
        cn_table.sort_values('weight', ascending=False, inplace=True)

    return cn_table
