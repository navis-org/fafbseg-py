======================================
Python Tools to Work with FAFB Autoseg
======================================

FAFBseg-py is a set of Python tools to work with neuron skeletons produced from
Google's `flood filling dataset <https://www.biorxiv.org/content/10.1101/605634v1>`_
hosted on a `CATMAID <https://catmaid.readthedocs.io/en/stable/>`_ server.


Quick Start
===========

Install the latest version of fafbseg-py with :code:`pip`.

.. code-block:: bat

   pip3 install git+git://github.com/flyconnectome/fafbseg-py.git

Then make sure to install required dependencies
`pymaid <https://pymaid.readthedocs.io/en/latest/>`_,
`brainmappy <https://github.com/schlegelp/brainmappy>`_,
`inquirer <https://magmax.org/python-inquirer/index.html>`_ and
`iPython <https://ipython.org/install.html>`_:

.. code-block:: bat

    pip3 install git+git://github.com/schlegelp/pymaid@master
    pip3 install git+git://github.com/schlegelp/brainmappy@master
    pip3 install inquirer
    pip3 install ipython

You will need brainmaps API access. See
`brainmappy <https://github.com/schlegelp/brainmappy>`_ for details.

Now have a look at the :doc:`examles<examples/index>` or the :doc:`API<api>`
to get started.

Contents
=========

.. toctree::
   :caption: Table of Contents
   :maxdepth: 1

   examples/index
   api
