.. _api:

API Documentation
=================

FAFBseg is divided into separate modules dedicated to a single
data source/type or functionality:

  - ``fafbseg.flywire`` for FlyWire-related functions
  - ``fafbseg.google`` for Google segmentation-related functions
  - ``fafbseg.synapses`` for querying Buhmann et al. synapse predictions
  - ``fafbseg.xform`` for transforming spatial data between FAFB14 and flywire's FAFB14.1
  - ``fafbseg.move`` for moving/merging data between data sets

See below for a by-module breakdown:

Flywire segmentation
--------------------
.. autosummary::
    :toctree: generated/

    fafbseg.flywire.locs_to_segments
    fafbseg.flywire.encode_url
    fafbseg.flywire.decode_url
    fafbseg.flywire.fetch_edit_history
    fafbseg.flywire.fetch_leaderboard
    fafbseg.flywire.locs_to_supervoxels
    fafbseg.flywire.skid_to_id
    fafbseg.flywire.update_ids
    fafbseg.flywire.get_mesh_neuron
    fafbseg.flywire.skeletonize_neuron
    fafbseg.flywire.generate_open_ends_url
    fafbseg.flywire.merge_flywire_neuron

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

    fafbseg.synapses.locs_to_segments
    fafbseg.synapses.query_synapses
    fafbseg.synapses.query_connections
    fafbseg.synapses.get_neuron_synapses
    fafbseg.synapses.get_neuron_synapses
    fafbseg.synapses.assign_connectors

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
