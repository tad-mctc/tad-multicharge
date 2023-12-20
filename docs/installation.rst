Installation
------------

pip
~~~

*tad-multicharge* can easily be installed with ``pip``.

.. code::

    pip install tad-multicharge


From source
~~~~~~~~~~~

This project is hosted on GitHub at `tad-mctc/tad-multicharge <https://github.com/tad-mctc/tad-multicharge>`__.
Obtain the source by cloning the repository with

.. code::

    git clone https://github.com/tad-mctc/tad-multicharge
    cd tad-multicharge

We recommend using a `conda <https://conda.io/>`__ environment to install the package.
You can setup the environment manager using a `mambaforge <https://github.com/conda-forge/miniforge>`__ installer.
Install the required dependencies from the conda-forge channel.

.. code::

    mamba env create -n torch -f environment.yaml
    mamba activate torch

Install this project with ``pip`` in the environment

.. code::

    pip install .

The following dependencies are required

- `numpy <https://numpy.org/>`__
- `tad_mctc <https://github.com/tad-mctc/tad_mctc/>`__
- `torch <https://pytorch.org/>`__
- `pytest <https://docs.pytest.org/>`__ (tests only)

Development
-----------

For development, additionally install the following tools in your environment.

.. code::

    mamba install black covdefaults mypy pre-commit pylint pytest pytest-cov pytest-xdist tox
    pip install pytest-random-order

With pip, add the option ``-e`` for installing in development mode, and add ``[dev]`` for the development dependencies

.. code::

    pip install -e .[dev]

The pre-commit hooks are initialized by running the following command in the root of the repository.

.. code::

    pre-commit install

For testing all Python environments, simply run `tox`.

.. code::

    tox

Note that this randomizes the order of tests but skips "large" tests. To modify this behavior, `tox` has to skip the optional *posargs*.

.. code::

    tox -- test
