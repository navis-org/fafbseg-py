[![Documentation Status](https://readthedocs.org/projects/fafbseg-py/badge/?version=latest)](https://fafbseg-py.readthedocs.io/en/latest/?badge=latest)

# FAFBseg
Tools to work with manually generated and auto-segmented data in FAFB.

## Documentation
FAFBseg is on [readthedocs](https://fafbseg-py.readthedocs.io/en/latest/).

## Quickstart
Because this is a private repository, installation is a bit more complicated
than usual. The easiest way to run this in a terminal:

```
pip3 install git+git://github.com/flyconnectome/fafbseg-py.git
```

To update an existing installation run this:

```
pip3 install git+git://github.com/flyconnectome/fafbseg-py.git --upgrade
```

## Requirements
You will need brainmaps API access. See [brainmappy](https://github.com/schlegelp/brainmappy)
for details.

On top of that, you need to install below dependencies.

### Dependencies
Make sure you have the *most_recent* version of the following libraries:

- [pymaid](https://pymaid.readthedocs.io/en/latest/): `pip3 install git+git://github.com/schlegelp/pymaid@master`
- [brainmappy](https://github.com/schlegelp/brainmappy): `pip3 install git+git://github.com/schlegelp/brainmappy@master`
- [inquirer](https://magmax.org/python-inquirer/index.html): `pip3 install inquirer`
- [iPython](https://ipython.org/install.html): `pip3 install ipython`
