# Data

## `global_area_ids.npy.zip` & `volume_name_dict.json`

These files were provided by Sven Dorkenwald. He used neuropil volumes
mapped from the JFRC2 template brain into FAFB14 space to assign each
synapse to a neuropil. `global_area_ids` is a list of `int8` - use
`volume_name_dict` to map to neuropil names. Unassignable synapses (~20M) have
id `-1`. 
