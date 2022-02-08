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

from .segmentation import neuron_to_segments

from ..synapses.utils import catmaid_table
from .. import spine

__all__ = ['fetch_synapses', 'fetch_connectivity']


def fetch_synapses(x, attach=True, dataset='production', progress=True):
    """Fetch Buhmann et al. (2019) synapses for given neuron(s).

    Uses a service on services.itanna.io hosted by Eric Perlman and Davi Bock.

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


def fetch_connectivity(x, segmentation='fafb-ffn1-20200412', clean=True, style='catmaid',
                       min_score=30, progress=True):
    """Fetch Buhmann et al. (2019) connectivity for given neuron(s).

    Uses a service on services.itanna.io hosted by Eric Perlman and Davi Bock.

    Parameters
    ----------
    x :             Neuron/List | int | list of int
                    Either a Google segment ID, a list thereof or a Neuron/List.
                    For neurons, the segmentation ID(s) will be mapped using the
                    node table.
    segmentation :  str
                    Which segmentation to search. Currently available:

                      - "fafb-ffn1-20200412" (default)
                      - "fafb-ffn1-20190805"
    clean :         bool
                    If True, we will perform some clean up of the connectivity
                    table:

    style :         "simple" | "catmaid"
                    Style of the returned table.

    Returns
    -------
    pd.DataFrame
                Connectivity table.

    Examples
    --------
    >>> import fafbseg
    >>> import pymaid
    >>> rm = pymaid.connect_catmaid()
    >>> # Load a neuron
    >>> n = pymaid.get_neurons(16)
    >>> # Get this neuron's synaptic partners
    >>> cn_table = fafbseg.google.fetch_connectivity(n)

    """
    # First we need to map the query to IDs
    if isinstance(x, (navis.TreeNeuron, navis.NeuronList)):
        # Get segments overlapping with these neurons
        overlaps = neuron_to_segments(x)

        ids = {}
        for n in overlaps.columns:
            this = overlaps[overlaps[n] > 0][n]
            ids.update(dict(zip(this.index.values, [n] * this.shape[0])))
    elif isinstance(x, (int, np.int)):
        ids = {x: x}
    else:
        ids = {n: n for n in navis.utils.make_iterable(x)}

    # Query the synapses
    syn = spine.synapses.get_connectivity(list(ids.keys()),
                                          segmentation=segmentation)

    # Next we need to run some clean-up:
    # 1. Drop below threshold connections
    if min_score:
        syn = syn[syn.cleft_scores >= min_score]
    # 2. Drop connections involving 0 (background, glia)
    syn = syn[(syn.pre != 0) & (syn.post != 0)]

    # Avoid copy warning
    syn = syn.copy()

    # Now map the Google IDs back to the IDs the came from (i.e. skeleton IDs)
    syn['pre'] = syn.pre.map(lambda x: ids.get(x, x))
    syn['post'] = syn.post.map(lambda x: ids.get(x, x))

    # Turn into connectivity table
    cn_table = syn.groupby(['pre', 'post'], as_index=False).size().to_frame().reset_index(drop=False).rename({0: 'weight'}, axis=1)

    # Style
    if style == 'catmaid':
        cn_table = catmaid_table(cn_table, list(set(ids.values())))
    else:
        cn_table.sort_values('weight', ascending=False, inplace=True)

    return cn_table
