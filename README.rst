Torch Autodiff Multicharge
==========================

.. image:: https://img.shields.io/badge/python-%3E=3.8-blue.svg
    :target: https://img.shields.io/badge/python-3.8%20|%203.9%20|%203.10%20|%203.11-blue.svg
    :alt: Python Versions

.. image:: https://img.shields.io/github/v/release/tad-mctc/tad-multicharge
    :target: https://github.com/tad-mctc/tad-multicharge/releases/latest
    :alt: Release

.. image:: https://img.shields.io/pypi/v/tad-multicharge
    :target: https://pypi.org/project/tad-multicharge/
    :alt: PyPI

.. image:: https://img.shields.io/badge/License-LGPL_v3-blue.svg
    :target: https://www.gnu.org/licenses/lgpl-3.0
    :alt: LGPL-3.0

.. image:: https://github.com/tad-mctc/tad-multicharge/actions/workflows/python.yaml/badge.svg
    :target: https://github.com/tad-mctc/tad-multicharge/actions/workflows/python.yaml
    :alt: CI

.. image:: https://readthedocs.org/projects/tad-multicharge/badge/?version=latest
    :target: https://tad-multicharge.readthedocs.io
    :alt: Documentation Status

.. image:: https://codecov.io/gh/tad-mctc/tad-multicharge/branch/main/graph/badge.svg?token=OGJJnZ6t4G
    :target: https://codecov.io/gh/tad-mctc/tad-multicharge
    :alt: Coverage

.. image:: https://results.pre-commit.ci/badge/github/tad-mctc/tad-multicharge/main.svg
    :target: https://results.pre-commit.ci/latest/github/tad-mctc/tad-multicharge/main
    :alt: pre-commit.ci status


PyTorch implementation of the electronegativity equilibration (EEQ) model for atomic partial charges.
This module allows to process a single structure or a batch of structures for the calculation of atom-resolved dispersion energies.

For details on the EEQ model, see

- \S. A. Ghasemi, A. Hofstetter, S. Saha, and S. Goedecker, *Phys. Rev. B*, **2015**, *92*, 045131. DOI: `10.1103/PhysRevB.92.045131 <https://doi.org/10.1103/PhysRevB.92.045131>`__

- \E. Caldeweyher, S. Ehlert, A. Hansen, H. Neugebauer, S. Spicher, C. Bannwarth and S. Grimme, *J. Chem. Phys.*, **2019**, *150*, 154122. DOI: `10.1063/1.5090222 <https://dx.doi.org/10.1063/1.5090222>`__


For alternative implementations, also check out

`multicharge <https://github.com/grimme-lab/multicharge>`__:
  Implementation of the EEQ model in Fortran.

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
- `tad-mctc <https://github.com/tad-mctc/tad-mctc/>`__
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

Examples
--------

The following example shows how to calculate the EEQ partial charges and the corresponding electrostatic energy for a single structure.

.. code:: python

    import torch
    from tad_multicharge import eeq

    numbers = torch.tensor([7, 7, 1, 1, 1, 1, 1, 1])

    # coordinates in Bohr
    positions = torch.tensor(
        [
            [-2.98334550857544, -0.08808205276728, +0.00000000000000],
            [+2.98334550857544, +0.08808205276728, +0.00000000000000],
            [-4.07920360565186, +0.25775116682053, +1.52985656261444],
            [-1.60526800155640, +1.24380481243134, +0.00000000000000],
            [-4.07920360565186, +0.25775116682053, -1.52985656261444],
            [+4.07920360565186, -0.25775116682053, -1.52985656261444],
            [+1.60526800155640, -1.24380481243134, +0.00000000000000],
            [+4.07920360565186, -0.25775116682053, +1.52985656261444],
        ]
    )

    total_charge = torch.tensor(0.0)
    cn = torch.tensor([3.0, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    eeq_model = eeq.EEQModel.param2019()
    energy, qat = eeq_model.solve(numbers, positions, total_charge, cn)

    print(torch.sum(energy, -1))
    # tensor(-0.1750)
    print(qat)
    # tensor([-0.8347, -0.8347,  0.2731,  0.2886,  0.2731,  0.2731,  0.2886,  0.2731])

The next example shows the calculation of the electrostatic energy with a simpler API for a batch of structures.

.. code:: python

    import torch
    from tad_multicharge import eeq
    from tad_mctc.batch import pack
    from tad_mctc.convert import symbol_to_number

    # S22 system 4: formamide dimer
    numbers = pack(
        (
            symbol_to_number("C C N N H H H H H H O O".split()),
            symbol_to_number("C O N H H H".split()),
        )
    )

    # coordinates in Bohr
    positions = pack(
        (
            torch.tensor(
                [
                    [-3.81469488143921, +0.09993441402912, 0.00000000000000],
                    [+3.81469488143921, -0.09993441402912, 0.00000000000000],
                    [-2.66030049324036, -2.15898251533508, 0.00000000000000],
                    [+2.66030049324036, +2.15898251533508, 0.00000000000000],
                    [-0.73178529739380, -2.28237795829773, 0.00000000000000],
                    [-5.89039325714111, -0.02589114569128, 0.00000000000000],
                    [-3.71254944801331, -3.73605775833130, 0.00000000000000],
                    [+3.71254944801331, +3.73605775833130, 0.00000000000000],
                    [+0.73178529739380, +2.28237795829773, 0.00000000000000],
                    [+5.89039325714111, +0.02589114569128, 0.00000000000000],
                    [-2.74426102638245, +2.16115570068359, 0.00000000000000],
                    [+2.74426102638245, -2.16115570068359, 0.00000000000000],
                ]
            ),
            torch.tensor(
                [
                    [-0.55569743203406, +1.09030425468557, 0.00000000000000],
                    [+0.51473634678469, +3.15152550263611, 0.00000000000000],
                    [+0.59869690244446, -1.16861263789477, 0.00000000000000],
                    [-0.45355203669134, -2.74568780438064, 0.00000000000000],
                    [+2.52721209544999, -1.29200800956867, 0.00000000000000],
                    [-2.63139587595376, +0.96447869452240, 0.00000000000000],
                ]
            ),
        )
    )

    # total charge of both system
    charge = torch.tensor([0.0, 0.0])

    # calculate electrostatic energy in Hartree
    energy = torch.sum(eeq.get_energy(numbers, positions, charge), -1)

    torch.set_printoptions(precision=10)
    print(energy)
    # tensor([-0.2086755037, -0.0972094536])
    print(energy[0] - 2 * energy[1])
    # tensor(-0.0142565966)


Contributing
------------

This is a volunteer open source projects and contributions are always welcome.
Please, take a moment to read the `contributing guidelines <CONTRIBUTING.md>`__.

License
-------

This project is free software: you can redistribute it and/or modify it under the terms of the Lesser GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This project is distributed in the hope that it will be useful, but without any warranty; without even the implied warranty of merchantability or fitness for a particular purpose. See the Lesser GNU General Public License for more details.

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in this project by you, as defined in the Lesser GNU General Public license, shall be licensed as above, without any additional terms or conditions.
