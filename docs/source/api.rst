.. _api:

API Documentation
=================

FAFBseg is divided into separate modules to split functions by data source/type:

  - ``fafbseg.flywire`` for FlyWire-related functions
  - ``fafbseg.google`` for Google segmentation-related functions
  - ``fafbseg.xform`` for transforming spatial data between FAFB14 and flywire's FAFB14.1
  - ``fafbseg.move`` for moving/merging data between data sets

See below for a by-module breakdown:

FlyWire segmentation
--------------------
.. autosummary::
    :toctree: generated/

    fafbseg.flywire.locs_to_segments
    fafbseg.flywire.neuron_to_segments
    fafbseg.flywire.encode_url
    fafbseg.flywire.decode_url
    fafbseg.flywire.fetch_edit_history
    fafbseg.flywire.fetch_leaderboard
    fafbseg.flywire.l2_info
    fafbseg.flywire.l2_graph
    fafbseg.flywire.l2_dotprops
    fafbseg.flywire.l2_skeleton
    fafbseg.flywire.locs_to_supervoxels
    fafbseg.flywire.skid_to_id
    fafbseg.flywire.is_latest_root
    fafbseg.flywire.update_ids
    fafbseg.flywire.get_mesh_neuron
    fafbseg.flywire.get_somas
    fafbseg.flywire.skeletonize_neuron
    fafbseg.flywire.skeletonize_neuron_parallel
    fafbseg.flywire.generate_open_ends_url
    fafbseg.flywire.merge_flywire_neuron

For interaction with the annotation/materialization engine:

.. autosummary::
    :toctree: generated/

    fafbseg.flywire.get_materialization_versions
    fafbseg.flywire.create_annotation_table
    fafbseg.flywire.get_annotation_tables
    fafbseg.flywire.get_annotation_table_info
    fafbseg.flywire.get_annotations
    fafbseg.flywire.delete_annotations
    fafbseg.flywire.upload_annotations

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

Buhmann synapse predictions
---------------------------
.. autosummary::
    :toctree: generated/

    fafbseg.google.synapses.fetch_connectivity
    fafbseg.flywire.synapses.fetch_adjacency
    fafbseg.flywire.synapses.fetch_connectivity
    fafbseg.flywire.synapses.fetch_synapses
    fafbseg.flywire.synapses.predict_transmitter
    fafbseg.synapses.plot_nt_predictions

Spatial transformation
----------------------
.. autosummary::
    :toctree: generated/

    fafbseg.xform.flywire_to_fafb14
    fafbseg.xform.fafb14_to_flywire

Merging/combining data
----------------------
.. autosummary::
    :toctree: generated/

    fafbseg.move.merge_into_catmaid
