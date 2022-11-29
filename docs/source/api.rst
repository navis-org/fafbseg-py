.. _api:

API Documentation
=================

FAFBseg is divided into separate modules to split functions by data source/type:

  - ``fafbseg.flywire`` for FlyWire-related functions
  - ``fafbseg.google`` for Google segmentation-related functions
  - ``fafbseg.xform`` for transforming spatial data between FAFB14 and FlyWire's FAFB14.1
  - ``fafbseg.move`` for moving/merging data between data sets

See below for a by-module breakdown.

FlyWire
-------

Interact with the segmentation:

.. autosummary::
    :toctree: generated/

    fafbseg.flywire.locs_to_segments
    fafbseg.flywire.neuron_to_segments
    fafbseg.flywire.encode_url
    fafbseg.flywire.decode_url
    fafbseg.flywire.locs_to_supervoxels
    fafbseg.flywire.supervoxels_to_roots
    fafbseg.flywire.skid_to_id
    fafbseg.flywire.is_latest_root
    fafbseg.flywire.update_ids
    fafbseg.flywire.get_voxels
    fafbseg.flywire.is_proofread
    fafbseg.flywire.find_common_time
    fafbseg.flywire.find_anchor_loc

Fetch neurons:

.. autosummary::
    :toctree: generated/

    fafbseg.flywire.get_mesh_neuron
    fafbseg.flywire.get_somas
    fafbseg.flywire.skeletonize_neuron
    fafbseg.flywire.skeletonize_neuron_parallel
    fafbseg.flywire.merge_flywire_neuron

L2 data:

.. autosummary::
    :toctree: generated/

    fafbseg.flywire.l2_info
    fafbseg.flywire.l2_graph
    fafbseg.flywire.l2_dotprops
    fafbseg.flywire.l2_skeleton

Misc:

.. autosummary::
    :toctree: generated/

    fafbseg.flywire.fetch_edit_history
    fafbseg.flywire.fetch_leaderboard
    fafbseg.flywire.get_neuropil_volumes
    fafbseg.flywire.get_lineage_graph
    fafbseg.flywire.get_lr_position
    fafbseg.flywire.get_voxels

For interaction with the annotation/materialization engine:

.. autosummary::
    :toctree: generated/

    fafbseg.flywire.get_materialization_versions
    fafbseg.flywire.create_annotation_table
    fafbseg.flywire.list_annotation_tables
    fafbseg.flywire.get_annotation_table_info
    fafbseg.flywire.get_annotations
    fafbseg.flywire.delete_annotations
    fafbseg.flywire.upload_annotations
    fafbseg.find_celltypes

Google segmentation
-------------------
.. autosummary::
    :toctree: generated/

    fafbseg.google.locs_to_segments
    fafbseg.google.segments_to_neuron
    fafbseg.google.segments_to_skids
    fafbseg.google.neuron_to_segments
    fafbseg.google.find_autoseg_fragments
    fafbseg.google.find_fragments
    fafbseg.google.find_missed_branches
    fafbseg.google.get_mesh
    fafbseg.google.autoreview_edges
    fafbseg.google.test_edges

Connectivity
------------
.. autosummary::
    :toctree: generated/

    fafbseg.google.synapses.fetch_connectivity
    fafbseg.flywire.synapses.fetch_adjacency
    fafbseg.flywire.synapses.fetch_connectivity
    fafbseg.flywire.synapses.fetch_synapses
    fafbseg.flywire.synapses.synapse_counts
    fafbseg.flywire.synapses.predict_transmitter
    fafbseg.synapses.plot_nt_predictions

Spatial transformation
----------------------
Note that typically you will want to use e.g.
``navis.xform_brain(data, source='FAFB14', target='FLYWIRE')`` but you can
also use these low-level functions:

.. autosummary::
    :toctree: generated/

    fafbseg.xform.flywire_to_fafb14
    fafbseg.xform.fafb14_to_flywire

Merging/combining data
----------------------
.. autosummary::
    :toctree: generated/

    fafbseg.move.merge_into_catmaid
