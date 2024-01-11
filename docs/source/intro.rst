.. _introduction:

Overview
========

What is FAFB?
-------------
The FAFB (short for "full adult fly brain") dataset is an serial-section transmission
electron microscopy (ssTEM) image data set of an entire female *Drosophila melanogaster*
brain imaged by Zhihao Zheng and Davi Bock
(see `Zheng et al., 2018 <https://www.sciencedirect.com/science/article/pii/S0092867418307876>`_).
The raw image data can be downloaded from https://temca2data.org/.

From around 2016, a consortium of labs *manually* reconstructed neurons and their connectivity
in this dataset using `CATMAID <https://catmaid.readthedocs.io/en/stable/>`_.
Published data from this effort is hosted by `Virtual Fly Brain <https://catmaid.virtualflybrain.org/>`_.
That manual reconstruction effort has now been superseded by the automatic
segmentations described below.

FAFB segmentations
------------------
Generally speaking, "segmentation" refers to the automatic labelling of some
feature in image data using machine learning algorithms. These features can be
neurons but also ultrastructural structures such as synapses.

FlyWire neuron segmentation
***************************
In mid 2020, the Seung and Murthy labs at Princeton made their
segmentation of FAFB public through `"FlyWire" <https://flywire.ai/>`_ and
many labs have since contributed to proofreading the dataset.

A first version of the proofread dataset was publicly released in July 2023 as
version "630". A second version with additional proofreading will be made
available in early 2024 as version "783".
See `Dorkenwald et al. <https://www.biorxiv.org/content/10.1101/2023.06.27.546656v2>`_
and `Schlegel et al. <https://www.biorxiv.org/content/10.1101/2023.06.27.546055v2>`_
for reference.

.. note::
    On a sidenote: the FAFB image data was realigned for the FlyWire segmentation.
    This means that x/y/z coordinates typically vary by a micron or so
    between original FAFB (also called "FAFB v14" or just "FAFB14") and FlyWire
    (also called "FAFB v14.1" or "FAFB14.1"). Keep that in mind when comparing data.

    Deformation fields mapping between FAFB14 and FlyWire have been kindly provided
    by the Seung lab and can be used via `fafbseg` (see tutorials).

Google neuron segmentation
**************************
In 2018/19, Peter Li (Google) auto-segmented the FAFB dataset using Google's flood
filling algorithm. See `this <http://fafb-ffn1.storage.googleapis.com/landing.html>`_
website for the paper, examples and data. Peter kindly shared skeletons derived
from the segmentation with the FAFB tracing community early on. These skeletons
were loaded into CATMAID instances by Tom Kazimiers and Eric Perlman.

Synaptic partner predictions
****************************
Building on their own segmentation of synaptic clefts in FAFB
(`Heinrich et al., 2018 <https://arxiv.org/abs/1805.02718>`_),
the Funke lab (Janelia Research Campus) produced a synaptic partners prediction
for FAFB (`Buhmann et al., 2019 <https://www.biorxiv.org/content/10.1101/2019.12.12.874172v2>`_).

This data is effectively represented by pairs of x/y/z coordinates where one
coordinate is pre- and the other one is postsynaptic. In combination with either
the Google or the FlyWire segmentation, we can map these connections onto
neurons to compile connectivity tables.

The raw data is publicly available: see
`this <https://github.com/funkelab/synful_fafb>`_ repository by the Funke lab.
To make it easier for researcher to access, Sven Dorkenwald, Forrest
Collman et al. have loaded the data into their CAVE annotation backend.

Bringing it all together
------------------------
As the introduction above shows, there are various types of FAFB data (skeletons,
meshes, segmentations, synapses, synaptic partners) available on various
platforms (CATMAID, FlyWire and other web services) and in two different
spaces (FAFB14 and FlyWire/FAFB14.1).

``FAFBseg`` and its R analog (`link <https://github.com/natverse/fafbseg>`_)
provide a single interface to draw from and combine all the FAFB data.

General layout
**************
Currently, ``FAFBseg`` is divided into two main modules:

- ``fafbseg.flywire``: work with FlyWire segmentation of FAFB
- ``fafbseg.google``: work with Google segmentation of FAFB

You will find that certain functions have a version for Google and for FlyWire.
For example, :func:`fafbseg.flywire.locs_to_segments` and
:func:`fafbseg.google.locs_to_segments` let you map locations to segmentation
IDs for the respective data set. That said: at this point the Google segmentation
has effectively been superseded by FlyWire and has therefore seen less
attention.

Please check out the :ref:`tutorials<tutorials>` for details!
