.. _api:

API Documentation
=================

FAFBseg is divided into separate modules to split functions by data source/type:

  - ``fafbseg.flywire`` for FlyWire functions
  - ``fafbseg.google`` for functions related to Google's segmentation of FAFB
  - ``fafbseg.xform`` for transforming spatial data between FAFB14 and FlyWire's FAFB14.1

See below for a by-module breakdown.

FlyWire
-------

Interact with the segmentation:

.. autosummary::
    :toctree: generated/

    fafbseg.flywire.locs_to_segments
    fafbseg.flywire.neuron_to_segments
    fafbseg.flywire.locs_to_supervoxels
    fafbseg.flywire.supervoxels_to_roots
    fafbseg.flywire.skid_to_id
    fafbseg.flywire.is_latest_root
    fafbseg.flywire.update_ids
    fafbseg.flywire.get_voxels
    fafbseg.flywire.is_proofread
    fafbseg.flywire.find_common_time
    fafbseg.flywire.find_anchor_loc
    fafbseg.flywire.get_segmentation_cutout

Fetch neurons:

.. autosummary::
    :toctree: generated/

    fafbseg.flywire.get_mesh_neuron
    fafbseg.flywire.get_somas
    fafbseg.flywire.get_skeletons
    fafbseg.flywire.skeletonize_neuron
    fafbseg.flywire.skeletonize_neuron_parallel

Connectivity:

.. autosummary::
    :toctree: generated/

    fafbseg.flywire.synapses.get_adjacency
    fafbseg.flywire.synapses.get_connectivity
    fafbseg.flywire.synapses.get_synapses
    fafbseg.flywire.synapses.get_synapse_counts
    fafbseg.flywire.synapses.get_transmitter_predictions

L2-related functions:

.. autosummary::
    :toctree: generated/

    fafbseg.flywire.get_l2_info
    fafbseg.flywire.get_l2_graph
    fafbseg.flywire.get_l2_dotprops
    fafbseg.flywire.get_l2_skeleton

Misc:

.. autosummary::
    :toctree: generated/

    fafbseg.flywire.get_edit_history
    fafbseg.flywire.get_leaderboard
    fafbseg.flywire.get_neuropil_volumes
    fafbseg.flywire.get_lineage_graph
    fafbseg.flywire.get_lr_position
    fafbseg.flywire.get_voxels

.. _api_annotations:

For fetching annotations:

.. autosummary::
    :toctree: generated/

    fafbseg.flywire.search_annotations
    fafbseg.flywire.search_community_annotations
    fafbseg.flywire.get_hierarchical_annotations
    fafbseg.flywire.NeuronCriteria

For interaction with the CAVE engine:

.. autosummary::
    :toctree: generated/

    fafbseg.flywire.get_materialization_versions
    fafbseg.flywire.create_cave_table
    fafbseg.flywire.list_cave_tables
    fafbseg.flywire.get_cave_table_info
    fafbseg.flywire.get_cave_table
    fafbseg.flywire.delete_annotations
    fafbseg.flywire.upload_annotations


Utility functions:

.. autosummary::
    :toctree: generated/

    fafbseg.flywire.set_default_dataset
    fafbseg.flywire.set_default_annotation_version
    fafbseg.flywire.get_user_information
    fafbseg.flywire.encode_url
    fafbseg.flywire.decode_url


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


Spatial transformation
----------------------
Note that typically you will want to use e.g.
``navis.xform_brain(data, source='FAFB14', target='FLYWIRE')`` but you can
also use these low-level functions:

.. autosummary::
    :toctree: generated/

    fafbseg.xform.flywire_to_fafb14
    fafbseg.xform.fafb14_to_flywire
