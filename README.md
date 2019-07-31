# fafbseg
Tools to work with manually generated and auto-segmented data in FAFB.

## Install
Because this is a private repository, installation is a bit more complicated
than usual. The easiest way is to first clone the repository:

```
git clone https://github.com/flyconnectome/fafbseg-py.git
```

Then CD into the directory and install the package in "edit" mode:

```
pip install -e .
```

This will simply create a pointer for Python to this repository. To update you
can now simply `git pull`.

## Dependencies
Make sure you have the *most_recent* version of the following libraries:

- [pymaid](https://pymaid.readthedocs.io/en/latest/): `pip3 install git+git://github.com/schlegelp/pymaid@master`
- [brainmappy](https://github.com/schlegelp/brainmappy): `pip3 install git+git://github.com/schlegelp/brainmappy@master`
- [inquirer](https://magmax.org/python-inquirer/index.html): `pip3 install inquirer`
- [iPython](https://ipython.org/install.html): `pip3 install ipython`

## Examples

### Merging a neuron from v14 autoseg into manual tracing
First fire up a terminal and start your interactive Python session by typing
`ipython`. Then run this code:

```Python
# Import relevant libraries
import fafbseg
import pymaid
import brainmappy as bm

# First connect to brainmaps (see brainmappy on how to get your credentials)
session = bm.acquire_credentials()

# Set the volume ID - make sure this is always the most recent/ best one
bm.set_global_volume('772153499790:fafb_v14:fafb-ffn1-20190521')

# Set up connections to the Catmaid instances
manual = pymaid.CatmaidInstance('https://neuropil.janelia.org/tracing/fafb/v14',
                                'HTTP_PW',
                                'HTTP_USER',
                                'API_TOKEN')

auto = pymaid.CatmaidInstance('https://neuropil.janelia.org/tracing/fafb/v14seg-Li-190411.0',
                              'HTTP_PW',
                              'HTTP_USER',
                              'API_TOKEN')

# Fetch the autoseg neuron to transfer to v14
x = pymaid.get_neuron(267355161, remote_instance=auto)

# Get the neuron's annotations so that they can be merged too
x.get_annotations(remote_instance=auto)

"""
At this point you could make changes to your neuron like:
 - remove some parts e.g. if you only want to import the axon
 - prune twigs to get rid of those annoying "bristles"
 - etc
"""

# Start the commit process (see video below for a demonstration)
resp = fafbseg.merge_neuron(x, target_instance=manual)
```

![merge example](https://github.com/flyconnectome/fafbseg-py/blob/master/media/screencast.gif?raw=true)

#### Merge finished - What now?
Success! The neuron has now been merged into existing manual tracings - what now?

**Minimally** you should have a look at the sites where existing and new
tracings were joined. The respective nodes will both be tagged
with `Joined from/into {SKELETON_ID}` and have a confidence of _1_ so that they are
easy to find in the treenode table:

![screeshot1](https://github.com/flyconnectome/fafbseg-py/blob/master/media/screenshot1.png?raw=true)

Depending on how much you care about the neuron, you want do a **full review**
to make sure that nothing was missed during the merge process.

#### Caveats
The merge procedure is a lengthy process and there is a chance that your local
data will diverge from the live CATMAID server (i.e. people make changes that
the script is unaware off). You should consider to:

- upload neurons in only small batches
- if possible make sure nobody is working on the neuron(s) you are merging into
- ideally run the merge when few people in CATMAID are tracing

#### Something went wrong - What now?
There are a few problems you might run into and that could cause the merging
procedure to stop. Generally speaking, the script is failsafe: e.g. if the
upload fails half-way through, you should be able to just restart and the
script will recognise changes that have already been made and skip these.

Especially if you are on slow connections, you should consider decreasing the
number of parallel requests allowed to lower the chances that something goes
wrong:

```
# Default is 100 -> let's lower that
manual.max_threads = 50
```
