.. _general_setup:

General setup
=============

This small tutorial explains the general setup you will have to run each time
you start a new Python session.

First fire up a terminal and start your interactive Python session:

.. code-block:: bat

  ipython


First we'll import the relevant libraries

.. code-block:: python

  import fafbseg
  import pymaid
  import brainmappy as bm


Next: setup brainmappy! If you are doing this for the *very first time* you
will need  a ``client_secret.json`` (see
`brainmappy <https://github.com/schlegelp/brainmappy>`_ for details) to
authenticate with your Google account:

.. code-block:: python

  # First time connecting to brainmaps
  session = bm.acquire_credentials('path/to/client_secret.json')


From now on credentials are stored locally and in the future you can simply run:

.. code-block:: python

  session = bm.acquire_credentials()


We also need to tell brainmaps which segmentation volume we are working with:

.. code-block:: python

  # Set the segmentation volume ID
  bm.set_global_volume('772153499790:fafb_v14:fafb-ffn1-20190805')

.. tip::

    Each CATMAID ``autoseg`` instance has data for a specific segmentation
    volume. You **have** to make sure that the volume ID set via
    ``bm.set_global_volume`` matches the segmentation used to generate the
    skeletons in that ``autoseg`` CATMAID instance.


Get connections to your ``manual`` and ``autoseg`` CATMAID set up:

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

You're all done and ready to get to work!

.. tip::

    ``ipython`` offers auto-completion: try for example typing in
    ``bm.set_`` and then hitting TAB. There is also a neat feature for repeating
    past commands: type in ``manual =`` and hit the up arrow on your keyboard
    to cycle through all past commands that match. This is very useful for
    re-occurring code like this general setup.
