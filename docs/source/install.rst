.. _installing:

Installation
============
To install the latest stable version of :code:`fafbseg-py` with :code:`pip`.

.. code-block:: bat

   pip3 install fafbseg

For the development version from Github:

.. code-block:: bat

   pip3 install git+https://github.com/navis-org/fafbseg-py.git


After the successful installation, please check out the
:ref:`FlyWire setup<flywire_setup>` for instructions on how to retrieve and
set your FlyWire API token.

To get started, have a look at the :ref:`examples<tutorials>` or the :ref:`API reference<api>`.


.. warning::

  For Windows users: ``fafbseg`` is unlikely to work off the bat as some of its
  dependencies might not install correctly. I highly recommend you install and use the
  `Windows Subsystem for Linux <https://learn.microsoft.com/en-us/windows/wsl/install>`_
  which lets you circumvent the problem by running a virtual Linux on your
  Windows machine.
