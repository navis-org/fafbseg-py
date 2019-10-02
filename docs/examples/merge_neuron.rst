.. _merge_example:

Merging
=======

This example will illustrate how to merge a neuron from v14 autoseg into
manual tracings:

.. note::

    For this to work, you need to have CATMAID API write and import access.

Please first see and follow the instructions in the
:doc:`General Setup<general_setup>`

OK now that we're all set, we can start the actual merging process:

Fetch the autoseg neuron to transfer to v14

.. code-block:: python

  x = pymaid.get_neuron(267355161, remote_instance=auto)

Get the neuron's annotations so that they can be merged too:

.. code-block:: python

  x.get_annotations(remote_instance=auto)

At this point you could make changes to your neuron like:
   - remove some parts e.g. if you only want to import the axon
   - prune twigs to get rid of those annoying "bristles"
   - ...

Next, start the actual commit process (see video below for a demonstration).
See :func:`fafbseg.merge_neuron` for additional parameters!

.. code-block:: python

  resp = fafbseg.merge_neuron(x, target_instance=manual, tag='YOURTAG')


Merge finished - What now?
--------------------------

Success! The neuron has now been merged into existing manual tracings - what now?

**Minimally** you should have a look at the sites where existing and new
tracings were joined. The respective nodes will both be tagged
with ``Joined from/into {SKELETON_ID}`` and have a confidence of ``1`` so that they are
easy to find in the treenode table:

.. image:: https://github.com/flyconnectome/fafbseg-py-media/blob/master/media/screenshot1.png?raw=true
   :width: 100%

Depending on how much you care about the neuron, you want do a **full review**
to make sure that nothing was missed during the merge process.

Caveats
-------

The merge procedure is a lengthy process and there is a chance that your local
data will diverge from the live CATMAID server (i.e. people make changes that
the script is unaware off). You should consider to:

- upload neurons in only small batches
- if possible make sure nobody is working on the neuron(s) you are merging into
- ideally run the merge when few people in CATMAID are tracing

Something went wrong - What now?
--------------------------------

There are a few problems you might run into and that could cause the merging
procedure to stop. Generally speaking, the script is failsafe: e.g. if the
upload fails half-way through, you should be able to just restart and the
script will recognise changes that have already been made and skip these.

Especially if you are on slow connections, you should consider decreasing the
number of parallel requests allowed to lower the chances that something goes
wrong:

.. code-block:: python

  # Default is 100 -> let's lower that
  manual.max_threads = 50
