.. _api:

API Documentation
=================

Mapping
-------
These functions will let you map between autoseg and manual CATMAID.

.. autosummary::
    :toctree: generated/

    ~fafbseg.search.segments_to_skids
    ~fafbseg.search.segments_to_neuron
    ~fafbseg.search.neuron_to_segments
    ~fafbseg.search.find_autoseg_fragments
    ~fafbseg.search.find_fragments

Merge
-----
These functions let you merge autoseg neurons into manual.

.. autosummary::
    :toctree: generated/

    ~fafbseg.merge.merge_neuron
    ~fafbseg.merge.find_missed_branches
