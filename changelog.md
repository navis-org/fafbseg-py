# Changelog

## [2.0.0] - 2023-10-09

This is a major release with lots of under-the-hood reworks to accompany the
public release of FlyWire.

### Querying the public release data
With the publication of the two FlyWire papers on bioRxiv everyone access to the
public released data - currently this corresponds to materialization version 630.

Previously, many `fafbseg.flywire` functions accepted a `dataset` parameter
that accepted (for the most part) either "production" or "sandbox". As of
version `2.0.0` "public" is also an accepted value:

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

The default for `dataset` is still "production" but this can be by either
calling `flywire.set_default_dataset` at the beginning of a session...

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

Corresponding functions such as `flywire.fetch_synapses` have had their defaults
changed to `filtered=True` and `min_score=None` which makes it so that this
filtered synapse table is queried by default. We recommend sticking to these defaults.

Importantly, querying the originally full synapse table requires access to the
production dataset and will not work with the "public" release version.

### Added

- `flywire.get_skeletons` downloads skeletons precomputed for the public (v630) release
- added functions to query the hierarchical annotations (Classification column in Codex):
  `flywire.search_annotations` and `flywire.get_hierarchical_annotations`
- added function to query the community annotations: `flywire.search_community_annotations`

### Breaking

- `flywire.get_annotation_tables` has been renamed to `flywire.list_annotation_tables`
- neuropils returned by synapse queries had their side (`_L`/`_R`) flipped to compensate
  for the inversion during image acquisition of the FAFB volume and now refer
  to the correct side from the flies perspective
- `flywire.encode_url` has been reworked and some of the parameters have been renamed
