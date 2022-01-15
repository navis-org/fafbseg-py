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

"""Functions to work with neurotransmitter predictions."""

import math
import navis

import seaborn as sns
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from collections import namedtuple


# transmitters + colors
trans = ['gaba', 'acetylcholine', 'glutamate', 'octopamine', 'serotonin',
         'dopamine']
colors = [(0.8352941176470589, 0.6588235294117647, 0.2823529411764706),
          (0.5843137254901961, 0.6392156862745098, 0.807843137254902),
          (0.5254901960784314, 0.6588235294117647, 0.34901960784313724),
          (0.4470588235294118, 0.3607843137254902, 0.596078431372549),
          (0.5490196078431373, 0.3843137254901961, 0.5843137254901961),
          (0.7215686274509804, 0.4745098039215686, 0.4117647058823529)]

prediction = namedtuple('prediction', ['transmitter', 'probability'])


def collapse_nt_predictions(pred, single_pred=False, weighted=True, id_col=None):
    """Collapses predictions using a weighted average.

    We use the `cleft_scores` as weights. Note that if there are no
    `cleft_scores` above 0 (happens sometimes with very small sets of synapses)
    the predictions will also be 0.

    Parameters
    ----------
    pred :          pd.DataFrame
                    Table with synapse neurotransmitter predictions.
    single_pred :   bool
                    If True, will return only the highest
    id_col :        str, optional
                    A column in `pred`. If provided, will collapse predictions
                    for every unique value in that column. Use this to point to
                    a column containing neuron IDs to get a prediction for
                    individual neurons.
    weighted :      bool
                    If True, will weight predictions based on confidence: higher
                    cleft score = more weight.

    Returns
    -------
    tuple
                    If ``single_pred=True`` and ``id_col=None`` return a tuple
                    of (transmitter, confidence) - e.g. ("acetylcholine", 0.89)

    dict
                    If ``single_pred=True`` and ``id_col!=None`` return a
                    dictionary mapping ids to predictions - e.g.
                    ``{"12345": ("acetylcholine", 0.89), "56789": ("gaba", 0.76)}``

    pd.Series
                    If ``single_pred=False`` and ``id_col=None`` return a pandas
                    Series::

                                        confidence
                       acetylcholine          0.89
                       gaba                   0.02
                       ...

    pd.DataFrame
                    If ``single_pred=False`` and ``id_col!=None`` return a
                    DataFrame::

                                             12345    56789
                       acetylcholine          0.89      0.2
                       gaba                   0.02     0.76
                       ...

    """

    if id_col is not None:
        # Collapse predictions for every unique value of id_col
        ids = pred[id_col].unique()
        res = {i: collapse_nt_predictions(pred[pred[id_col] == i],
                                          single_pred=single_pred,
                                          weighted=weighted) for i in ids}

        # Combine results
        if not single_pred:
            # Label series and combine
            for r in res:
                res[r].name = r

            res = pd.concat(list(res.values()), join='outer', axis=1).fillna(0)

        return res

    # Drop NAs (some synapses have no prediction)
    pred = pred[pred[trans].any(axis=1)]

    if pred.empty:
        raise ValueError('No synapses with transmitter predictions.')

    # Generate series with confidence for all transmitters
    if weighted:
        weight_col = 'cleft_scores' if 'cleft_scores' in pred.columns else 'cleft_score'
        if pred[weight_col].sum() > 0:
            pred_weight = pd.Series({t: np.average(pred[t], weights=pred[weight_col])
                                     for t in trans}, name='confidence')
        else:
            pred_weight = pd.Series({t: 0 for t in trans}, name='confidence')
    else:
        pred_weight = pd.Series({t: pred[t].mean() for t in trans})

    if single_pred:
        # Get the highest predicted transmitter
        top_ix = np.argmax(pred_weight)
        return prediction(pred_weight.index[top_ix], pred_weight.iloc[top_ix])

    return pred_weight


def plot_nt_predictions(pred, bins=20, id_col=None, ax=None, legend=True, **kwargs):
    """Plot neurotransmitter predictions.

    Parameters
    ----------
    pred :      pandas.DataFrame
                Table with synapse neurotransmitter predictions.
                For example as returned from
                ``fafbseg.flywire.fetch_synapses(..., transmitters=True)``.
    bins :      int | sequence of scalars
                Either number of bins or a sequence of bin edges.
    id_col :    str, optional
                Name of a column in `pred`. If provided, will generate one
                subplot for every unique neuron ID in that column.
    ax :        matplotlib ax, optional
                If not provided will plot on a new axis. Ignored when ``id_col``
                is provided.
    **kwargs
                Keyword arguments are passed on to ``plt.subplots``.

    Returns
    -------
    ax

    """
    if id_col:
        # Unique values in that column
        uni = pred[id_col].unique()

        nrows = ncols = len(uni) ** 0.5

        ncols = math.ceil(ncols)
        if (int(nrows) * ncols) >= len(uni):
            nrows = int(nrows)
        else:
            nrows = math.ceil(nrows)

        # Generate figure
        fig, axes = plt.subplots(nrows, ncols, **kwargs)

        # Flatten axes
        if nrows > 1 and ncols > 1:
            axes = [a for l in axes for a in l]

        for v, ax in zip(uni, axes):
            _ = plot_nt_predictions(pred[pred[id_col] == v], ax=ax, bins=bins,
                                    id_col=None)
            ax.set_title(v)

        return axes

    if not navis.utils.is_iterable(bins):
        bins = np.linspace(0, pred.cleft_score.max(), bins)

    grp = pred.groupby(pd.cut(pred.cleft_score, bins, right=False))[trans]
    mn = grp.mean()
    sem = grp.sem()

    if not ax:
        fig, ax = plt.subplots(**kwargs)

    x = [ix.left + (ix.right - ix.left) / 2 for ix in mn.index]
    for t, c in zip(trans, colors):
        ax.plot(x, mn[t].values, c=c, label=t, lw=1.5)
        ax.fill_between(x,
                        mn[t].values + sem[t],
                        mn[t].values - sem[t],
                        color=c, alpha=.2)

    ax.set_xlabel('cleft score')
    ax.set_ylabel('mean confidence')

    if legend:
        ax.legend()

    return ax
