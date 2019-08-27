.. _general_setup:

General setup
=============

This small tutorial explains the general setup you will have to run each time
your start a new Python session.

First fire up a terminal and start your interactive Python session:

.. code-block:: bat

  ipython

First we'll import the relevant libraries

.. code-block:: python

  import fafbseg
  import pymaid
  import brainmappy as bm

Next, setup brainmappy (see `brainmappy <https://github.com/schlegelp/brainmappy>`_
on how to acquire credentials)

.. code-block:: python

  # First connect to brainmaps
  session = bm.acquire_credentials()

  # Set the segmentation volume ID
  bm.set_global_volume('772153499790:fafb_v14:fafb-ffn1-20190805')

Get your CATMAID instances with manual and autoseg skeletons set up.

.. code-block:: python

  manual = pymaid.CatmaidInstance('https://neuropil.janelia.org/tracing/fafb/v14',
                                  'HTTP_USER',
                                  'HTTP_PW',
                                  'API_TOKEN')

.. code-block:: python

  auto = pymaid.CatmaidInstance('https://neuropil.janelia.org/tracing/fafb/v14-seg-li-190805.0',
                                'HTTP_USER',
                                'HTTP_PW',
                                'API_TOKEN')

.. tip::

    Make sure that the volume ID set via ``bm.set_global_volume`` matches
    the segmentation used to generate the skeletons in your ``auto``
    CATMAID instance.

By default, pymaid caches data it receives from the CATMAID server. In this use
case, however, we do not want to lag behind the server if at all possible. Let's
disable caching

.. code-block:: python

  manual.caching = False
  auto.caching = False

Depending on your internet connection and the performances of the servers,
you encounter ``HTTPErrors`` of various kinds. In that case, you will have
to lower the number of parallel queries before running your code

.. code-block:: python

  # Default is 100 parallel threads
  manual.max_threads = 20
  auto.max_threads = 20


You're all done and ready to get to work!
