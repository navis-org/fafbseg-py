.. _introduction:

Overview
========

What is FAFB?
-------------
The FAFB dataset is an SSTEM image data set of an entire female flybrain
imaged by Zhihao Zheng and Davi Bock
(`Zheng et al., 2018 <https://www.sciencedirect.com/science/article/pii/S0092867418307876>`_).
The raw image data can be downloaded from https://temca2data.org/.

A consortium of labs has been *manually* reconstructing neurons and their connectivity
in this data set since 2016 using `CATMAID <https://catmaid.readthedocs.io/en/stable/>`_.
Published data from this effort is hosted by `Virtual Fly Brain <https://catmaid.virtualflybrain.org/>`_.
Unpublished data from ongoing work is hosted in a "walled garden" CATMAID instance.
Labs have to apply to join this community and agree to guidelines that ensure a
fair and respectful treatment of each other's unpublished data.

FAFB segmentations
------------------
Generally speaking, "segmentation" refers to the automatic labelling of some feature in
image data using machine learning algorithms. These features can be neurons but
also ultrastructural structures such as synapses.

Google neuron segmentation
**************************
In 2018/19, Peter Li auto-segmented the FAFB dataset using Google's flood
filling algorithm. See `this <http://fafb-ffn1.storage.googleapis.com/landing.html>`_
website for the paper, examples and data. Peter kindly shared skeletons derived
from the segmentation with the FAFB tracing community early on. These skeletons
were loaded into CATMAID instances by Tom Kazimiers and Eric Perlman.

To summarize, there are currently 5 "walled garden" FAFB CATMAID instances:

1. The main instance at ``fafb/v14/`` (neuropil)
2. The first Google segmentation at ``fafb/v14-seg`` (neuropil)
3. A second iteration of the Google segmentation at ``fafb/v14seg-Li-190411.0`` (neuropil)
4. A third iteration of the Google segmentation at ``fafb/v14-seg-li-190805.0`` (neuropil)
5. The most recent iteration of the Google segmentation at ``catmaid/fafb-v14-seg-li-200412.0/`` (spine)

FlyWire neuron segmentation
***************************
As of mid 2020, the Seung and Murthy labs at Princeton have made their
segmentation of FAFB public as `"FlyWire" <https://flywire.ai/>`_.

Anyone can join this project but community guidelines similar to those for the
"walled garden" apply. Importantly, newly joined users will be confined to a
"sandbox" dataset for training and only be admitted to the production dataset
after passing a quick test.

Important to note is that the FAFB image data was realigned for the FlyWire
segmentation. This is why x/y/z coordinates typically vary by a micron or so
between original FAFB (also called "FAFB v14" or just "FAFB14") and FlyWire
(also called "FAFB v14.1" or "FAFB14.1").

Deformation fields mapping between FAFB14 and FlyWire have been kindly provided
by the Seung lab.

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
However, to make it easier for researcher to access, Sven Dorkenwald, Forrest
Collman et al. have loaded the data into their annotation backend.

Bringing it all together
------------------------
As the introduction above shows, there are various types of FAFB data (skeletons,
meshes, segmentations, synapses, synaptic partners) available on various
platforms (CATMAID, FlyWire and other web services) and in two different
spaces (FAFB14 and FlyWire/FAFB14.1).

``FAFBseg`` and its R analog (`link <https://github.com/natverse/fafbseg>`_)
provide a single interface to draw from and combine all the FAFB data.

In particular, it enables you to:

- load FlyWire neurons (e.g. for visualization and analysis)
- skeletonize FlyWire neurons
- query connectivity for FlyWire or FAFB14 neurons
- transform data between FAFB14 and FlyWire
- move data between manual FAFB14, Google and FlyWire segmentation (experimental and subject to community guidelines!)

General layout
**************
Currently, ``FAFBseg`` is divided into three main modules:

- ``fafbseg.flywire``: work with FlyWire segmentation
- ``fafbseg.google``: work with Google segmentation of FAFB14
- ``fafbseg.move``: import/export/move data around

You will find that certain functions have a version for Google and for FlyWire.
For example, :func:`fafbseg.flywire.locs_to_segments` and
:func:`fafbseg.google.locs_to_segments` let you map locations to segmentation
IDs for the respective data set.

Check out the tutorials for details!
