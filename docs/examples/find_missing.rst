.. _missing_example:

Finding missed branches
=======================

This examples will illustrate how to use the ``auto-seg`` to find potential
missed branches.

.. note::

    For this to work, you need to have CATMAID API write access.

Please first see and follow the instructions in the
:doc:`General Setup<general_setup>`

OK now that we're all set, we can start the actual merging process:

In this example, we will take a single manually traced neuron and try
finding potentially missed branches using the ``auto-seg``.

First fetch the neuron (exchange the ``16`` for the skeleton ID
of your neuron):

.. code-block:: python

  x = pymaid.get_neuron(16, remote_instance=manual)

Now find and tag missed branches (see:func:`fafbseg.find_missed_branches` for
additional parameters):

.. code-block:: python

  (summary,
   fragments,
   branches) = fafbseg.find_missed_branches(x, autoseg_instance=auto)

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
