[![Documentation Status](https://readthedocs.org/projects/fafbseg-py/badge/?version=latest)](https://fafbseg-py.readthedocs.io/en/latest/?badge=latest) [![Tests](https://github.com/navis-org/fafbseg-py/actions/workflows/test-package.yml/badge.svg)](https://github.com/navis-org/fafbseg-py/actions/workflows/test-package.yml) [![DOI](https://zenodo.org/badge/197735091.svg)](https://zenodo.org/badge/latestdoi/197735091)

<img src="https://github.com/navis-org/fafbseg-py/blob/master/docs/_static/logo2.png?raw=true" height="60">

Tools to work with the [FlyWire](https://flywire.ai/) and [Google](https://fafb-ffn1.storage.googleapis.com/landing.html) segmentations of the FAFB EM dataset. Fully interoperable with [navis](https://github.com/navis-org/navis).

## Features
Here are just some of the things you can do with ``fafbseg``:

* map locations or supervoxels to segmentation IDs
* load neuron meshes and skeletons
* generate high quality neuron meshes and skeletons from scratch
* query connectivity and annotations
* parse and generate FlyWire neuroglancer URLs
* transform neurons from FAFB/FlyWire space to other brains spaces (e.g. hemibrain)

## Documentation
FAFBseg is on [readthedocs](https://fafbseg-py.readthedocs.io/en/latest/).

## Quickstart
Install latest stable version

```bash
pip3 install fafbseg -U
```

Install from Github
```bash
pip3 install git+https://github.com/flyconnectome/fafbseg-py.git
```

## How to cite
If you use `fafbseg` for your publication, please cite the two FlyWire papers:

1. "_Whole-brain annotation and multi-connectome cell typing quantifies circuit stereotypy in Drosophila_" Schlegel _et al._, bioRxiv (2023); doi: https://doi.org/10.1101/2023.06.27.546055
2. "_Neuronal wiring diagram of an adult brain_" Dorkenwald _et al._, bioRxiv (2023); doi: https://doi.org/10.1101/2023.06.27.546656

