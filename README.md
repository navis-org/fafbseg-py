# fafbseg
Tools to work with manually generated and auto-segmented data in FAFB.

## Dependencies
- [pymaid](https://pymaid.readthedocs.io/en/latest/)
- [brainmappy](https://github.com/schlegelp/brainmappy)

## Examples

Taking a neuron from v14 autoseg and importing it into manual tracing:
```Python
import fafbseg
import pymaid
import brainmappy as bm

# First connect to brainmaps (see brainmappy on how to get your credentials)
session = bm.acquire_credentials()

# Set the volume ID - make sure this is always the most recent/ best one
bm.set_global_volume('772153499790:fafb_v14:fafb_v14_16nm_v00c_split3xfill2')

# Set up connections to the Catmaid instances
manual = pymaid.CatmaidInstance('https://neuropil.janelia.org/tracing/fafb/v14',
                                'HTTP_PW',
                                'HTTP_USER',
                                'API_TOKEN')

auto = pymaid.CatmaidInstance('https://neuropil.janelia.org/tracing/fafb/v14seg-Li-190411.0',
                              'HTTP_PW',
                              'HTTP_USER',
                              'API_TOKEN')

# Fetch the autoseg neuron to transfer to manual
x = pymaid.get_neuron(267355161, remote_instance=auto)

# Start the commit process (see video below for more)
resp = fafbseg.merge_neuron(x, target_instance=manual)
```
