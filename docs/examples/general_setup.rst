.. _general_setup:

General setup
=============

This brief tutorial explains the general setup you will have to run each time
you start a new Python session.

First fire up a terminal and start your interactive Python session:

.. code-block:: bat

  ipython


First we'll import the relevant libraries

.. code-block:: python

  import fafbseg
  import pymaid

Next get connections to the ``manual`` and ``autoseg`` CATMAID instances set up
(make sure to replace ``HTTP_USER``, ``HTTP_PW`` and ``API_TOKEN`` with
the corresponding credentials):

.. code-block:: python

  manual = pymaid.CatmaidInstance('https://neuropil.janelia.org/tracing/fafb/v14',
                                  'HTTP_USER',
                                  'HTTP_PW',
                                  'API_TOKEN',
                                  caching=False,
                                  max_threads=20)

.. code-block:: python

  auto = pymaid.CatmaidInstance('https://neuropil.janelia.org/tracing/fafb/v14-seg-li-190805.0',
                                'HTTP_USER',
                                'HTTP_PW',
                                'API_TOKEN',
                                caching=False,
                                max_threads=20)

Choosing a segmentation source
------------------------------

When importing neurons into another CATMAID instance, ``fafbseg`` will suggest
potentially overlapping neurons for merging. For this it uses the
segmentation data from Google to determine if skeletons are actually in the same
"segment" or just adjacent.

There are three options for where to get that data from:

.. list-table::
    :widths: 12 20 10 10
    :header-rows: 1

    * - **Source**
      - **Description**
      - **Advantages**
      - **Disadvantage**
    * - 1. Google storage
      - Google has put their segmentation data on their cloud storage and we can
        use `CloudVolume <https://github.com/seung-lab/cloud-volume>`_ from the
        Seung lab to query it.
      - does not need any special permissions
      - slow
    * - 2. brainmaps API
      - This is the the same API that neuroglancer uses to browse the segmentation
        data.
      - blazingly fast
      - needs permission to access the brainmaps API (see
        `brainmappy <https://github.com/schlegelp/brainmappy>`_ for details)
    * - 3. Self-hosted remote data
      - The highest resolution (i.e. not downsampled) segmentation data for FAFB
        is ~850Gb. Given that the segmentation data is publicly available, you
        download and host it yourself - using for example
        `CloudVolumeServer <https://github.com/flyconnectome/CloudVolumeServer>`_.
      - potentially faster than brainmaps
      - requires server & know-how


Using Google Storage
********************

This is the easiest solution as it does not need special permission. To set it
up run :func:`fafbseg.use_google_storage` at start up:

.. code-block:: python

  # Accessing the most recent autoseg data
  fafbseg.use_google_storage("https://storage.googleapis.com/fafb-ffn1-20190805/segmentation")


Using brainmaps
***************

You will need the `brainmappy <https://github.com/schlegelp/brainmappy>_`
library for this. If you haven't already installed it, run this in a terminal:

.. code-block:: bat

    pip3 install git+git://github.com/schlegelp/brainmappy@master

To tell ``fafbseg`` to use brainmaps to query segmentation data use
:func:`fafbseg.use_brainmaps` (see
`brainmappy <https://github.com/schlegelp/brainmappy>`_ for explanation
on credentials).

If you are doing this for the very first time you also need to provide a
``client_secret.json`` file:

.. code-block:: python

  fafbseg.use_brainmaps('772153499790:fafb_v14:fafb-ffn1-20190805',
                        client_secret='path/to/client_secret.json')

From now on credentials are stored locally and in the future you can simply run:

.. code-block:: python

  fafbseg.use_google_storage('772153499790:fafb_v14:fafb-ffn1-20190805')

.. tip::

    Each CATMAID ``autoseg`` instance contains data for a specific segmentation
    volume. You **have** to make sure that the volume set via
    ``fafseg.use_...`` matches the segmentation used to generate the
    skeletons in that ``autoseg`` CATMAID instance.

Using self-hosted remote solution
*********************************

If you are self-hosting the data, you will need to pass a URL
to :func:`fafbseg.use_remote_service`. The service behind the URL has to
accept a list of x/y/z locations as POST and return a list of segmentation IDs
in the same order:

.. code-block:: python

  fafbseg.use_remote_service('https://my-server.com/seg/values')

Alternatively, set an environment variable:

.. code-block:: bat

  EXPORT SEG_ID_URL="https://my-server.com/seg/values"

If you have an environment variable set, you an simply run:

.. code-block:: python

  fafbseg.use_remote_service()


If you have set up one of the above explained means to access the segmentation
data, you're all done and ready to get to work!

.. tip::

    ``ipython`` offers auto-completion: try for example typing in
    ``fafbseg.use_`` and then hitting TAB. There is also a neat feature for
    repeating past commands: type in ``manual =`` and hit the up arrow on your
    keyboard to cycle through all past commands that match. This is very useful
    for re-occurring code like this general setup.
