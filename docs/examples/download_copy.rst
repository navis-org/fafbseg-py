.. _localcopy:

Downloading segmentation data
=============================

The segmentation data is publicly available. Instead of using remote service
(e.g. brainmaps) you can download it and query the local copy of the data.

For this to work well, you will need fast SSD drives (internal are best,
external USB drives work too albeit slower). The highest resolution version of
the segmentation data is ~850Gb so make sure your drives have enough free space.

Here is a brief explanation on how to download the data:

1. Install `gsutil <https://cloud.google.com/storage/docs/gsutil_install>`_ (no
   need to set up a Google Cloud account).
2. Open a terminal, navigate to your SSD and generate the folders:

.. code-block:: bat

  mkdir segmentation
  cd segmentation
  mkdir 8.0x8.0x40.0

3. Start the download (this may take a few hours depending on your internet
   connection):

.. code-block:: bat

  gsutil -m cp -r gs://fafb-ffn1-20190805/segmentation/info .
  gsutil -m cp -r gs://fafb-ffn1-20190805/segmentation/8.0x8.0x40.0/* 8.0x8.0x40.0/

4. Finally we need to modify the ``info`` file to reflect the fact that we
   only downloaded the highest resolution: we need to open that file and replace
   its contents. There are multiple ways of doing this but this should work
   for most users:

.. code-block:: bat

  cp info info.bkp
  open info -e

This should have opened a text editor. Replace the contents with the below
configuration, save and close the ``info``:

.. code-block:: bat

  {
   "@type" : "neuroglancer_multiscale_volume",
   "data_type" : "uint64",
   "num_channels" : 1,
   "scales" : [
      {
         "chunk_sizes" : [
            [ 128, 128, 32 ]
         ],
         "compressed_segmentation_block_size" : [ 16, 16, 2 ],
         "encoding" : "compressed_segmentation",
         "key" : "8.0x8.0x40.0",
         "resolution" : [ 8, 8, 40 ],
         "sharding" : {
            "@type" : "neuroglancer_uint64_sharded_v1",
            "data_encoding" : "gzip",
            "hash" : "identity",
            "minishard_bits" : 6,
            "minishard_index_encoding" : "gzip",
            "preshift_bits" : 9,
            "shard_bits" : 13
         },
         "size" : [ 124416, 67072, 7063 ]
      }
   ],
   "type" : "segmentation",
   "mesh": "mesh"
  }
