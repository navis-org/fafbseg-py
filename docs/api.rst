.. _api:

API Documentation
=================

Mapping
-------
These functions will let you map between autoseg and manual CATMAID.

.. autosummary::
    :toctree: generated/

    ~fafbseg.segments_to_skids
    ~fafbseg.segments_to_neuron
    ~fafbseg.neuron_to_segments
    ~fafbseg.find_autoseg_fragments
    ~fafbseg.find_fragments

Merge
-----
These functions let you merge autoseg neurons into manual.

.. autosummary::
    :toctree: generated/

    ~fafbseg.merge_neuron
    ~fafbseg.find_missed_branches
