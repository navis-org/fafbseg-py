.. _merging:

Merging
=======

This example will illustrate how to merge a neuron from v14 autoseg into
manual tracing:

First fire up a terminal and start your interactive Python session:

.. code-block:: bat

  ipython

Now we need to run some setup:

.. code-block:: python

  # Import relevant libraries
  import fafbseg
  import pymaid
  import brainmappy as bm

  # First connect to brainmaps (see brainmappy on how to get your credentials)
  session = bm.acquire_credentials()

  # Set the volume ID - make sure this is always the most recent/ best one
  bm.set_global_volume('772153499790:fafb_v14:fafb-ffn1-20190805')

  # Set up connections to the Catmaid instances
  manual = pymaid.CatmaidInstance('https://neuropil.janelia.org/tracing/fafb/v14',
                                  'HTTP_USER',
                                  'HTTP_PW',
                                  'API_TOKEN')

  # Please note that in almost all situations you would want to have the global
  # volume ID set above to be the same as used for skeletons in this autoseg
  # CATMAID instance
  auto = pymaid.CatmaidInstance('https://neuropil.janelia.org/tracing/fafb/v14-seg-Li-190805.0',
                                'HTTP_USER',
                                'HTTP_PW',
                                'API_TOKEN')

.. tip::

    Make sure that the volume ID set via ``bm.acquire_credentials`` matches
    the segmentation used to generate the skeletons in your ``auto``
    CATMAID instance.                           

OK, we're all set. Now we can start the actual merging process:

.. code-block:: python

  # Fetch the autoseg neuron to transfer to v14
  x = pymaid.get_neuron(267355161, remote_instance=auto)

  # Get the neuron's annotations so that they can be merged too
  x.get_annotations(remote_instance=auto)

  """
  At this point you could make changes to your neuron like:
   - remove some parts e.g. if you only want to import the axon
   - prune twigs to get rid of those annoying "bristles"
   - etc
  """

  # Start the commit process (see video below for a demonstration)
  resp = fafbseg.merge_neuron(x, target_instance=manual)


.. image:: https://github.com/flyconnectome/fafbseg-py/blob/master/media/screencast.gif?raw=true
   :width: 100%

Merge finished - What now?
--------------------------

Success! The neuron has now been merged into existing manual tracings - what now?

**Minimally** you should have a look at the sites where existing and new
tracings were joined. The respective nodes will both be tagged
with ``Joined from/into {SKELETON_ID}`` and have a confidence of ``1`` so that they are
easy to find in the treenode table:

.. image:: https://github.com/flyconnectome/fafbseg-py/blob/master/media/screenshot1.png?raw=true
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
