.. _flywire_setup:

FlyWire setup
=============

In this tutorial you will learn all you need to know to get started
querying the FlyWire dataset.

Setting the FlyWire secret
--------------------------
Most data queries against FlyWire use ``cloudvolume`` or ``caveclient`` which
provide interfaces with FlyWire's chunkedgraph and annotation backends,
respectively. To be able to make those query you need to generate and store
an API token (or "secret"). This needs to be done only once.

Generate your secret
********************
Go to https://globalv1.flywire-daf.com/auth/api/v1/user/token and log in with
the account you use for FlyWire. You should then get bit of text with a token
that looks something like this:

.. code-block:: bat

  "ghd2nfk67kdndncf5kdmsqwo8kf23md6"

That's your token!

Saving the secret
*****************
Your token must then be saved to a file on your computer:
``~/.cloudvolume/secrets/global.daf-apis.com-cave-secret.json``. That file's
content needs to look something like this:

.. code-block:: bat

  {
  "token: ""ghd2nfk67kdndncf5kdmsqwo8kf23md6"
  }

``fafbseg`` offers a convenient function that does that for you:

.. code-block:: python

  >>> from fafbseg import flywire
  >>> flywire.set_chunkedgraph_secret("ghd2nfk67kdndncf5kdmsqwo8kf23md6")

That's it, you're done! You should now be able to query the FlyWire dataset.


FlyWire datasets
----------------

FlyWire actually has three different datasets/versions:

1. The "Public release" is the static snapshot made available alongside the preprints
   and corresponds to materialization version ``630`` (see below for an explanation
   of materializations). Anyone has access to this dataset.
2. The "Sandbox" is a training ground that has seen minimal proofreading (i.e.
   is close to the bsae segmentation). Anyone has access to this dataset.
3. The "Production" dataset is where people do the actual proofreading/annotation.
   As such it is ahead of the public release dataset. To get access to the
   production dataset you have to approved by one of the community managers.

Most functions in ``fafbseg.flywire`` accept a ``dataset`` parameter. If not
specified it will fall back to the production dataset.

.. code-block:: python

  >>> from fafbseg import flywire
  >>> # Defaults to production
  >>> flywire.supervoxels_to_roots(79801523353597754)
  array([720575940631274967])
  >>> flywire.supervoxels_to_roots(79801523353597754, dataset='public')
  array([720575940621675174])


You can change this default by running this at the beginning of each session:

.. code-block:: python

  >>> from fafbseg import flywire
  >>> flywire.set_default_dataset('public')

See the docstring for :func:`~fafbseg.flywire.set_default_dataset` for details.

Alternatively, you can also set an ``FLYWIRE_DEFAULT_DATASET`` environment
variable *before* starting the Python session.

.. code-block:: bash

  $ export FLYWIRE_DEFAULT_DATASET="public"
  $ python


Environment variables can be set permanently too. The details of that depend
on your operating system and on which terminal (e.g. bash or zsh) you are using.
A quick Google should tell you how it works.


Understanding FlyWire root IDs
------------------------------

Under the hood FlyWire is using chunkedgraph, an octree-like structure, to manage
the segmentation. In brief, "supervoxels" are the atomic unit of the
segmentation which are grouped into "root IDs". Or conversely: each root ID is a
collection of supervoxels. Any edit to the segmentation is effectively
just the addition or subtraction of supervoxels.

Like supervoxels, root IDs are immutable though. So whenever edits are made
new root IDs are generated which then represent the post-edit agglomeration of
supervoxels. For example, splitting a neuron will generate two new root IDs
and invalidate its current root ID. Merging two neurons, on the other hand, will
invalidate the two old root IDs and generate one new root ID representing the
combination of their supervoxels.

Importantly, "outdated" root IDs are not deleted and you can still e.g. pull up
their meshes in the FlyWire neuroglancer. This is super convenient but it comes
with a caveat: you can find yourself with a list of root IDs that never
co-existed which can be problematic when querying associated meta data (see
paragraph below).

Here are a couple ``fabseg`` functions that will help you tracking root IDs:

.. autosummary::
    :toctree: generated/

    fafbseg.flywire.locs_to_segments
    fafbseg.flywire.locs_to_supervoxels
    fafbseg.flywire.supervoxels_to_roots
    fafbseg.flywire.is_latest_root
    fafbseg.flywire.update_ids
    fafbseg.flywire.find_common_time

Understanding materializations
------------------------------

As established above, root IDs can change over time. So how do we maintain the
link between a neuron and its meta data (e.g. its annotations, synapses, etc)
as it evolves? Principally this is done by associating each annotation with an
x/y/z coordinate. That coordinate maps to a supervoxel and we can then ask
which root ID it belongs to - or belonged to if we want to go back in time.

This kind of location to root ID look-up becomes rather expensive when working
with large tables: the (filtered) synapse table, for example, has 130M rows each
with a pre- and a postsynaptic x/y/z coordinate that needs to be mapped to a
root ID.

So
