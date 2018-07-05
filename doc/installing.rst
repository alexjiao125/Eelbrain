**********
Installing
**********

.. note::
    Eelbrain comes with a C extension. Precompiled binaries are currently
    provided for macOS and Windows through ``conda``. On other platforms,
    installing Eelbrain requires compilation.


.. contents:: Contents
   :local:


Installing with Anaconda (recommended)
--------------------------------------

For using Eelbrain with Continuum Analytics' `Anaconda
<https://www.anaconda.com/download/>`_, a `Conda environment
<https://conda.io/docs/user-guide/tasks/manage-environments.html>`_ can be
generated for Eelbrain with a single command::

    $ conda create -n eelbrain -c defaults -c conda-forge -c christianbrodbeck eelbrain

To add the required channels permanently::

    $ conda config --append channels conda-forge
    $ conda config --append channels christianbrodbeck

To install Eelbrain with its dependencies in an existing environment::

    $ conda install eelbrain

To update an existing Eelbrain installation::

    $ conda update eelbrain


Optional extensions
^^^^^^^^^^^^^^^^^^^

Eelbrain provides functions to interface with R if `rpy2
<http://rpy.sourceforge.net>`_ is installed::

    $ conda install rpy2


Installing from PYPI
--------------------

.. note::
   If you are setting up a new Python environment, the easier option is
   `Anaconda <https://store.continuum.io/cshop/anaconda/>`_.

Eelbrain can be installed from the
`Python Package Index (PYPI) <https://pypi.python.org/pypi/eelbrain>`_::

    $ pip install eelbrain

This will not install those dependencies that require compilation, since it
might be easier to install them from a different source. Eelbrain can be
installed with all its dependencies from the PYPI with::

    $ pip install eelbrain[full]

And, if you want to use pysurfer::

    $ pip install eelbrain[plot.brain]

Later, update Eelbrain with::

    $ pip install -U --upgrade-strategy only-if-needed eelbrain


`WxPython <http://www.wxpython.org>`_ is not strictily necessary but enhances
plots with several GUI elements (and enables the epoch rejection GUI).
wxPython `can not be installed from the PYPI
<http://stackoverflow.com/q/477573/166700>`_, but installers are provided
`here <http://www.wxpython.org/download.php>`__.


Optional extensions
^^^^^^^^^^^^^^^^^^^

The following items provide additional functionality if they are installed:

* Eelbrain provides functions to interface with R if `rpy2
  <http://rpy.sourceforge.net>`_ is installed: First install
  `R <http://www.r-project.org>`_, then use pip to install ``rpy2``:
  ``$ pip install rpy2``.
* A working `LaTeX <http://www.latex-project.org/>`_ installation (enables
  exporting tables as pdf).
