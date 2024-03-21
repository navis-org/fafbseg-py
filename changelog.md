# Changelog

## [3.0.3] - 2024-03-24
- make `is_proofread` play nicely with the public dataset (uses a different table for proofread neuron IDs)
- add option to `get_cave_table` to fill in user info (name + affiliation)
- bug fixes in `get_somas`, `find_mat_version` , `is_latest_root` and `is_valid_root`

## [3.0.2] - 2024-03-11
Fixes a bug in `search_annotations` when there are no cached versions.

## [3.0.1] - 2024-03-11
Small bug fix release.

1. We now prevent searches against live materialization in the public dataset.
2. Make get `get_user_information` robust against missing IDs.

## [3.0.0] - 2024-01-10
The default dataset is now the "public" dataset!

This release allows you to work with annotations (classes, types, etc) instead of
having to shuttle root IDs in and out of `fafbseg` functions.

Annotation queries will download (and cache) data from https://github.com/flyconnectome/flywire_annotations
which is currently principally based on Schlegel et al. (2023). We are planning
to keep updating these annotations in the future.

We also renamed a bunch of functions to make things more consistent (see
section on **Breaking changes**).

### Working with annotations
With this version, `fafbseg` introduces a `NeuronCriteria` class. It works
similar to that in `neuprint-python` and can be passed to many functions that
previously only accepted root IDs:

```python
>>> from fafbseg import flywire
>>> NC = flywire.NeuronCriteria
>>> flywire.search_annotations(NC(type='ps009', side='left'))
           supervoxel_id             root_id   ...  side  nerve
0      78886867319895457  720575940620322497   ...  left     CV
>>> flywire.get_connectivity(NC(type='ps009', side='left'))
                     pre                post  weight
0     720575940635179359  720575940620322497     181
1     720575940631446855  720575940620322497     125
2     720575940608118283  720575940620322497     109
```

Please see the updated annotation [tutorial](https://fafbseg-py.readthedocs.io/en/latest/source/gallery.html)
for details.

### Breaking changes
- prior to version `3.0.0`, some functions accepted a materialization version as
  `mat` and some as `materialization` argument; with this version this argument
  is consistently named `materialization`; this mostly affects
  connectivity-related functions
- the following functions have been renamed to avoid confusion:
  - `list_annotation_tables()` -> `list_cave_tables()`
  - `get_annotations()` -> `get_cave_table()`
  - `create_annotation_table()` -> `create_cave_table()`
  - `get_annotation_table_info()` -> `get_cave_table_info()`
- the following function have been renamed for consistency:
  - `fetch_synapses()` -> `get_synapses()`
  - `fetch_adjacency()` -> `get_adjacency()`
  - `fetch_connectivity()` -> `get_connectivity()`
  - `synapse_counts()` -> `get_synapse_counts()`
  - `predict_transmitter()` -> `get_transmitter_predictions`
  - `fetch_leaderboard()` -> `get_leaderboard()`
  - `fetch_edit_history()` -> `get_edit_history()`
  - `l2_graph` -> `get_l2_graph`
  - `l2_skeleton` -> `get_l2_skeleton`
  - `l2_dotprops` -> `get_l2_dotprops`
  - `l2_info` -> `get_l2_info`

### Other fixes/improvemnts
- you can now query the flat `783` segmentation by using `dataset="flat_783"`
- precomputed high-res skeletons for `783` have been made available; see
  `flywire.get_skeletons`

## [2.0.3] - 2023-11-29
One fix, one improvement:
- `flywire.fetch_connectivity` now allows `neuropils=True` to return edges broken
  down by neuropils
- fix `flywire.predict_neuropils` with `single_pred=True`

## [2.0.2] - 2023-11-14
Various fixes and improvements:
- `flywire.get_mesh_neuron` ignore `lod` parameter if not applicable
- `flywire.get_skeletons` now correctly loads radii information (requires
  up-to-date version of `navis`)
- `flywire.update_ids` is now much faster if `timestamp` is provided
- fix live connectivity queries
- improved error messages for connectivity queries

## [2.0.1] - 2023-10-11
Minor fixes and improvements:
- better handling of the CAVE secret
- improve `flywire.search_annotations`

## [2.0.0] - 2023-10-09
This is a major release with lots of under-the-hood reworks to accompany the
public release of FlyWire.

### Querying the public release data
With the publication of the two FlyWire papers on bioRxiv everyone access has to the
publicly released data - currently this corresponds to materialization version `630`.

Previously, many `fafbseg.flywire` functions already had a `dataset` parameter
that accepted either "production" or "sandbox". As of version `2.0.0` "public"
is also an accepted value:

```python
>>> from fafbseg import flywire
>>> flywire.fetch_connectivity(720575940622670174, dataset='public')
Using materialization version 630

                     pre                post  weight
0     720575940622670174  720575940626637002      65
1     720575940626637002  720575940622670174      60
2     720575940628529896  720575940622670174      41
3     720575940622670174  720575940631383400      41
4     720575940622670174  720575940620266689      39
...                  ...                 ...     ...
3453  720575940622670174  720575940547279839       1
3454  720575940622670174  720575940547426271       1
3455  720575940622670174  720575940547445727       1
3456  720575940622670174  720575940548226581       1
3457  720575940622670174  720575940538297971       1
```

The default for `dataset` is still "production" but this can be changed by either
calling [`flywire.set_default_dataset`](https://fafbseg-py.readthedocs.io/en/latest/source/generated/fafbseg.flywire.set_default_dataset.html) at the beginning of a session...

```python
>>> flywire.set_default_dataset('public')
```

... or by setting an environment variable `FLYWIRE_DEFAULT_DATASET="public"`.

The public data has a few idiosyncrasies, most of which we tried to abstract
away. Some, however, remain: for example synapse/connectivity queries are
more limited against the public data stack (see below).

### Synapse/connectivity queries
Prior to `v2.0.0` synapses and connectivity were fetched by querying the full
CAVE synapse table. For connectivity analyses in the FlyWire papers,
Sven Dorkenwald generated a synapse table (technically a "view") where he
deduplicated synapses and applied a higher confidence (cleft) score threshold of 50.
See [here](https://prod.flywire-daf.com/annotation/views/aligned_volume/fafb_seung_alignment_v0/table/valid_synapses_nt_v2)
for a full explanation (requires login with FlyWire account).

Corresponding functions such as [`flywire.fetch_synapses`](https://fafbseg-py.readthedocs.io/en/latest/source/generated/fafbseg.flywire.fetch_synapses.html) have had their defaults
changed to `filtered=True` and `min_score=None` which makes it so that this
filtered synapse table is queried by default. We recommend sticking to these defaults.

Importantly, querying the originally full synapse table requires access to the
production dataset and will not work with the "public" release version.

### Added

- [`flywire.get_skeletons`](https://fafbseg-py.readthedocs.io/en/latest/source/generated/fafbseg.flywire.get_skeletons.html) downloads skeletons precomputed for the public (v630) release
- added functions to query the hierarchical annotations (Classification column in Codex):
  [`flywire.search_annotations`](https://fafbseg-py.readthedocs.io/en/latest/source/generated/fafbseg.flywire.search_annotations.html) and [`flywire.get_hierarchical_annotations`](https://fafbseg-py.readthedocs.io/en/latest/source/generated/fafbseg.flywire.get_hierarchical_annotations.html)
- added function to query the community annotations: [`flywire.search_community_annotations`](https://fafbseg-py.readthedocs.io/en/latest/source/generated/fafbseg.flywire.search_community_annotations.html)

### Breaking

- `flywire.get_annotation_tables` has been renamed to [`flywire.list_annotation_tables`](https://fafbseg-py.readthedocs.io/en/latest/source/generated/fafbseg.flywire.list_annotation_tables.html)
- neuropils returned by synapse queries had their side (`_L`/`_R`) flipped to compensate
  for the inversion during image acquisition of the FAFB volume and now refer
  to the correct side from the flies perspective
- [`flywire.encode_url`](https://fafbseg-py.readthedocs.io/en/latest/source/generated/fafbseg.flywire.encode_url.html) has been reworked and some of the parameters have been renamed
