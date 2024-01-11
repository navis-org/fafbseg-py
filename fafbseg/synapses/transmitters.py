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

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from collections import namedtuple


# transmitters + colors
trans = ["gaba", "acetylcholine", "glutamate", "octopamine", "serotonin", "dopamine"]
colors = [
    (0.8352941176470589, 0.6588235294117647, 0.2823529411764706),
    (0.5843137254901961, 0.6392156862745098, 0.807843137254902),
    (0.5254901960784314, 0.6588235294117647, 0.34901960784313724),
    (0.4470588235294118, 0.3607843137254902, 0.596078431372549),
    (0.5490196078431373, 0.3843137254901961, 0.5843137254901961),
    (0.7215686274509804, 0.4745098039215686, 0.4117647058823529),
]

prediction = namedtuple("prediction", ["transmitter", "probability"])


def collapse_nt_predictions(
    pred, single_pred=False, weighted=True, id_col=None, raise_empty=True
):
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
    # Drop NAs (some synapses have no prediction)
    pred = pred[pred[trans].any(axis=1)]

    if pred.empty:
        if raise_empty:
            raise ValueError("No synapses with transmitter predictions.")
        elif not single_pred:
            return pd.DataFrame(None, index=trans)
        elif id_col:
            return {}
        else:
            return prediction(None, None)

    # If we have an ID column we have to group by those IDs
    if id_col is not None:
        ids = pred[id_col].unique()

        if weighted:
            weight_col = (
                "cleft_scores" if "cleft_scores" in pred.columns else "cleft_score"
            )
            res = pred.groupby(id_col).apply(
                lambda x: [np.average(x[t], weights=x[weight_col]) for t in trans]
            )
            res = pd.DataFrame(np.vstack(res.values), columns=trans, index=res.index)
        else:
            res = pred.groupby(id_col)[trans].mean()

        res = res.reindex(ids).fillna(0)

        # If no single prediction, just return the frame
        if not single_pred:
            return res.T

        top_nt = np.array(trans)[np.argmax(res.values, axis=1)]
        conf = np.max(res.values, axis=1)
        return {i: prediction(nt, conf) for i, nt, conf in zip(ids, top_nt, conf)}

    # If no predictions
    if pred.empty:
        if raise_empty:
            raise ValueError("No synapses with transmitter predictions.")
        elif single_pred:
            return prediction(None, None)
        else:
            return pd.DataFrame(None, index=trans)

    # Generate series with confidence for all transmitters
    if weighted:
        weight_col = "cleft_scores" if "cleft_scores" in pred.columns else "cleft_score"
        if pred[weight_col].sum() > 0:
            pred_weight = pd.Series(
                {t: np.average(pred[t], weights=pred[weight_col]) for t in trans},
                name="confidence",
            )
        else:
            pred_weight = pd.Series({t: 0 for t in trans}, name="confidence")
    else:
        pred_weight = pd.Series({t: pred[t].mean() for t in trans})

    if single_pred:
        # Get the highest predicted transmitter
        top_ix = np.argmax(pred_weight)
        return prediction(pred_weight.index[top_ix], pred_weight.iloc[top_ix])
    else:
        return pred_weight


def plot_nt_predictions(pred, bins=20, id_col=None, ax=None, legend=True, **kwargs):
    """Plot neurotransmitter predictions.

    Parameters
    ----------
    pred :      pandas.DataFrame
                Table with synapse neurotransmitter predictions.
                For example as returned from
                ``fafbseg.flywire.get_synapses(..., transmitters=True)``.
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
            _ = plot_nt_predictions(
                pred[pred[id_col] == v], ax=ax, bins=bins, id_col=None
            )
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
        ax.fill_between(
            x, mn[t].values + sem[t], mn[t].values - sem[t], color=c, alpha=0.2
        )

    ax.set_xlabel("cleft score")
    ax.set_ylabel("mean confidence")

    if legend:
        ax.legend()

    return ax


def paint_neuron(x, pred, max_dist=None):
    """Paint neuron by its neurotransmitter.

    For each vertex / skeleton node we ask what the closest transmitter is.

    Parameters
    ----------
    x :         TreeNeuron | MeshNeuron
    pred :      pandas.DataFrame
                Table with synapse neurotransmitter predictions.
                For example as returned from
                ``fafbseg.flywire.get_synapses(..., transmitters=True)``.

    Returns
    -------
    None
                For ``TreeNeuron``: adds a `transmitter` column to the node
                table. For ``MeshNeuron`` adds an array as `.transmitter`
                property.

    Examples
    --------
    >>> import navis                                            # doctest: +SKIP
    >>> from fafbseg import flywire, synapses                   # doctest: +SKIP
    >>> m = flywire.get_mesh_neuron(720575940608508996)         # doctest: +SKIP
    >>> x = m.skeleton                                          # doctest: +SKIP
    >>> pred = flywire.get_synapses(720575940608508996,         # doctest: +SKIP
    ...                             post=False, pre=True,       # doctest: +SKIP
    ...                             transmitters=True)          # doctest: +SKIP
    >>> synapses.paint_neuron(x, pred)                          # doctest: +SKIP
    >>> colors = {'gaba': (0.8352941176470589, 0.6588235294117647, 0.2823529411764706),
    ...           'acetylcholine': (0.5843137254901961, 0.6392156862745098, 0.807843137254902),
    ...           'glutamate': (0.5254901960784314, 0.6588235294117647, 0.34901960784313724),
    ...           'octopamine': (0.4470588235294118, 0.3607843137254902, 0.596078431372549),
    ...           'serotonin': (0.5490196078431373, 0.3843137254901961, 0.5843137254901961),
    ...           'dopamine': (0.7215686274509804, 0.4745098039215686, 0.4117647058823529)}
    >>> navis.plot3d(x, color_by='transmitter', palette=colors) # doctest: +SKIP

    """
    # Snap to nodes/vertices
    ix, _ = x.snap(pred[["pre_x", "pre_y", "pre_z"]].values)
    tr = np.array(trans)[np.argmax(pred[trans].values, axis=1)]
    tr_dict = dict(zip(ix, tr))

    # Get the geodesic distance from all nodes/vertices to synapse bearing
    # nodes/vertices
    dist = navis.geodesic_matrix(x, from_=ix)
    closest = dist.index[np.argmin(dist.values, axis=0)]
    closest_tr = np.array([tr_dict[i] for i in closest])

    if max_dist:
        closest_tr[dist.min(axis=0) > max_dist] = "NA"

    if isinstance(x, navis.TreeNeuron):
        x.nodes["transmitter"] = x.nodes.node_id.map(
            dict(zip(dist.columns, closest_tr))
        )
    else:
        x.transmitter = closest_tr
