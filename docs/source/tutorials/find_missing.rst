.. _missing_example:

Finding missed branches
=======================

This examples will illustrate how to use the Google segmentation to find
potential missed branches.

We will take a single, manually traced neuron and try finding potentially missed
branches by comparing it to the same Google skeleton.

First initialize connections to the ``manual`` and ``autoseg`` CATMAID instances
set up - make sure to replace ``HTTP_USER``, ``HTTP_PW`` and ``API_TOKEN`` with
your corresponding credentials:

.. code-block:: python

  manual = pymaid.CatmaidInstance('https://neuropil.janelia.org/tracing/fafb/v14',
                                  api_token='API_TOKEN',
                                  http_user='HTTP_USER',
                                  http_password='HTTP_PW',
                                  caching=False,
                                  max_threads=20)

.. code-block:: python

  auto = pymaid.CatmaidInstance('https://spine.janelia.org/catmaid/fafb-v14-seg-li-200412.0',
                                api_token='API_TOKEN',
                                http_user='HTTP_USER',
                                http_password='HTTP_PW',
                                caching=False,
                                max_threads=20)

Then fetch the neuron (exchange the ``16`` for the skeleton ID of your neuron):

.. code-block:: python

  x = pymaid.get_neuron(16, remote_instance=manual)

Now find and tag missed branches (see:func:`fafbseg.google.find_missed_branches` for
additional parameters):

.. code-block:: python

  (summary,
   fragments,
   branches) = fafbseg.google.find_missed_branches(x, autoseg_instance=auto)

Show summary of missed branches:

.. code-block:: python

  summary.head()

Co-visualize your neuron and potentially overlapping autoseg fragments:

.. code-block:: python

  x.plot3d(color='w')
  fragments.plot3d()

Visualize the potentially missed branches:

.. code-block:: python

  pymaid.clear3d()
  x.plot3d(color='w')
  branches.plot3d(color='r')
