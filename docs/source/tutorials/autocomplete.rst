.. _autocomplete:

Autocomplete CATMAID Neurons
============================

This examples will illustrate how to use the Google ``autoseg`` to autocomplete
a partially reconstructed CATMAID neuron.

.. note::

    For this to work, you need to have CATMAID API write and import access

First initialize connections to the ``manual`` and ``autoseg`` CATMAID instances
set up - make sure to replace ``HTTP_USER``, ``HTTP_PW`` and ``API_TOKEN`` with
your corresponding credentials:

.. code-block:: python

  import pymaid
  manual = pymaid.CatmaidInstance('https://neuropil.janelia.org/tracing/fafb/v14',
                                  api_token='API_TOKEN',
                                  http_user='HTTP_USER',
                                  http_password='HTTP_PW',
                                  caching=False,
                                  max_threads=20)

.. code-block:: python

  auto = pymaid.CatmaidInstance('https://spine.itanna.io/catmaid/fafb-v14-seg-li-200412.0',
                                api_token='API_TOKEN',
                                http_user='HTTP_USER',
                                http_password='HTTP_PW',
                                caching=False,
                                max_threads=20)

Next get the neuron to autocomplete (exchange the ``16`` for the skeleton ID
of your neuron).

.. code-block:: python

  x = pymaid.get_neuron(16, remote_instance=manual)


Now get everything that overlaps with this neuron in ``autoseg``. See
:func:`fafbseg.google.find_autoseg_fragments` on how to fine tune this step.

.. code-block:: python

  import fafbseg
  ol = fafbseg.google.find_autoseg_fragments(x, autoseg_instance=auto)


Now visualise the large overlapping fragments alongside the manual tracings:

.. code-block:: python

  x.plot3d(color='w')
  ol[ol.n_nodes > 40].plot3d()


You can use the 3D viewer to select which fragments actually overlap with
you neuron. For this, turn on the legend by pressing ``L`` (this might take a
second) and then pressing ``P`` to enable picking. Now you can click on the
legend entries to show/hide neurons.

Hide autoseg fragments you do **not** want to use to autocomplete your neuron
and then run this:

.. code-block:: python

  import numpy as np
  visible = ol[np.isin(ol.skeleton_id, navis.get_viewer().visible)]

Before we can start the merge process, we have to stitch all ``autoseg``
fragments to form a single virtual neuron for upload:

.. code-block:: python

  y = navis.stitch_neurons(visible, method='NONE')

If you want to have a final look this is how you can co-visualize the manual
tracings and the to-be-merged ``autoseg`` fragments:

.. code-block:: python

  x.plot3d(color='w', clear=True)
  y.plot3d(color='r')

Once you are ready start the upload process as described in
:doc:`Merging<merge_neuron>`. (see also :func:`fafbseg.move.merge_neuron` for
additional parameters):

.. code-block:: python

  resp = fafbseg.move.merge_neuron(y, target_instance=manual, tag='YOURTAG')


Gotchas
-------

When looking for overlapping ``autoseg`` fragments, you can end up finding the
autoseg version of your original neuron - ``x`` in above example. This happens
if somebody has merged a Google skeleton into ``x``.

This is problematic because ``fafbseg`` uses the skeleton ID to identify where
new and old nodes originate from but now we have two neurons with the same
skeleton ID. :func:`~fafbseg.merge_neuron` will throw in error::

  ValueError: Duplicate skeleton IDs found. Try manually assigning unique skeleton IDs.

To resolve this, you need to manually change the skeleton ID of ``y`` - ideally
to that of the Google fragment that got merged into it: look for an annotation
like ``Merged: Google: 5819659900`` and then change the skeleton ID::

  y.skeleton_id = '5819659900'
  y._clear_temp_attributes()
