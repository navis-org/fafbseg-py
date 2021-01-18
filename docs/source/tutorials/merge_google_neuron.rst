.. _merge_google_example:

Merging Skeletons from the Google segmentation into main CATMAID
================================================================
There are multiple CATMAID instances hosting skeletons made from various
iterations of the FAFB segmentation by Peter Li from Google. In this example,
you will learn how to fetch a skeleton from one of those instances and merge
it into existing (manual) tracings in the main FAFB v14 CATMAID instance.
We will focus on the most recent ``20200412.0`` but you can use any other
CATMAID instance.

.. note::

    For this to work, you need to have CATMAID API write and import access. If
    at any point you get a "You lack permissions" error, please get in touch
    with an administrator.

I *highly* recommend you try this out with a small chunk of neuron to begin with
as there is the potential to mess things up in the main CATMAID instance.
Ideally, you have somebody who is familiar with the process show you how it's
done!


Setting things up
-----------------
First fire up a terminal and start your interactive Python session:

.. code-block:: bat

  ipython


Then we import the relevant libraries:

.. code-block:: python

  import fafbseg
  import pymaid

Next get connections to the ``manual`` and ``autoseg`` CATMAID instances set up
- make sure to replace ``HTTP_USER``, ``HTTP_PW`` and ``API_TOKEN`` with
the corresponding credentials:

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

Grab the neuron you want to import
----------------------------------
OK now that we're all set, we need to grab the neuron you wish to import from
the CATMAID autoseg instance.

.. code-block:: python

  # 267355161 is the skeleton ID of the neuron
  x = pymaid.get_neuron(267355161, remote_instance=auto)

Get the neuron's annotations so that they can be merged too:

.. code-block:: python

  x.get_annotations(remote_instance=auto)

At this point you could make changes to your neuron like:
   - remove some parts e.g. if you only want to import the axon
   - prune twigs to get rid of those annoying "bristles"
   - ...

All of this is done with ``pymaid`` - if you are not yet familiar with it
it's worth getting at least a basic understanding of it! See the
`docs <https://pymaid.readthedocs.io/en/latest/>`_.

Merge the neuron
----------------
Next, start the actual commit process (see video below for a demonstration).
See :func:`fafbseg.move.merge_neuron` for additional parameters!

.. code-block:: python

  resp = fafbseg.move.merge_into_catmaid(x, target_instance=manual, tag='YOURTAG')

The ``YOURTAG`` should be something that identifies you and/or your group. It
will be added as annotation to the neuron after merging.

The above command will go through a sequence of data collecting and then
present you with candidates you might want to merge your neuron into. Carefully
read the instructions printed in the terminal at each step!

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

- upload neurons in only small batches (i.e. one at a time)
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
  manual.max_threads = 20
