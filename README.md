# Torch Autodiff Multicharge

<table>
  <tr>
    <td>Compatibility:</td>
    <td>
      <img src="https://img.shields.io/badge/Python-3.8%20|%203.9%20|%203.10%20|%203.11%20|%203.12-blue.svg" alt="Python Versions"/>
      <img src="https://img.shields.io/badge/PyTorch-%3E=1.11.0-blue.svg" alt="PyTorch Versions"/>
    </td>
  </tr>
  <tr>
    <td>Availability:</td>
    <td>
      <a href="https://github.com/tad-mctc/tad-multicharge/releases/latest">
        <img src="https://img.shields.io/github/v/release/tad-mctc/tad-multicharge?color=orange" alt="Release"/>
      </a>
      <a href="https://pypi.org/project/tad-multicharge/">
        <img src="https://img.shields.io/pypi/v/tad-multicharge?color=orange" alt="PyPI"/>
      </a>
      <a href="https://anaconda.org/conda-forge/tad-multicharge">
        <img src="https://img.shields.io/conda/vn/conda-forge/tad-multicharge.svg" alt="Conda Version"/>
      </a>
      <a href="http://www.apache.org/licenses/LICENSE-2.0">
        <img src="https://img.shields.io/badge/License-Apache%202.0-orange.svg" alt="Apache-2.0"/>
      </a>
    </td>
  </tr>
  <tr>
    <td>Status:</td>
    <td>
      <a href="https://github.com/tad-mctc/tad-multicharge/actions/workflows/ubuntu.yaml">
        <img src="https://github.com/tad-mctc/tad-multicharge/actions/workflows/ubuntu.yaml/badge.svg" alt="Test Status Ubuntu"/>
      </a>
      <a href="https://github.com/tad-mctc/tad-multicharge/actions/workflows/macos-x86.yaml">
        <img src="https://github.com/tad-mctc/tad-multicharge/actions/workflows/macos-x86.yaml/badge.svg" alt="Test Status macOS (x86)"/>
      </a>
      <a href="https://github.com/tad-mctc/tad-multicharge/actions/workflows/macos-arm.yaml">
        <img src="https://github.com/tad-mctc/tad-multicharge/actions/workflows/macos-arm.yaml/badge.svg" alt="Test Status macOS (ARM)"/>
      </a>
      <a href="https://github.com/tad-mctc/tad-multicharge/actions/workflows/windows.yaml">
        <img src="https://github.com/tad-mctc/tad-multicharge/actions/workflows/windows.yaml/badge.svg" alt="Test Status Windows"/>
      </a>
      <a href="https://github.com/tad-mctc/tad-multicharge/actions/workflows/release.yaml">
        <img src="https://github.com/tad-mctc/tad-multicharge/actions/workflows/release.yaml/badge.svg" alt="Build Status"/>
      </a>
      <a href="https://tad-multicharge.readthedocs.io">
        <img src="https://readthedocs.org/projects/tad-multicharge/badge/?version=latest" alt="Documentation Status"/>
      </a>
      <a href="https://results.pre-commit.ci/latest/github/tad-mctc/tad-multicharge/main">
        <img src="https://results.pre-commit.ci/badge/github/tad-mctc/tad-multicharge/main.svg" alt="pre-commit.ci Status"/>
      </a>
      <a href="https://codecov.io/gh/tad-mctc/tad-multicharge">
        <img src="https://codecov.io/gh/tad-mctc/tad-multicharge/branch/main/graph/badge.svg?token=OGJJnZ6t4G" alt="Coverage"/>
      </a>
    </td>
  </tr>
</table>

<br>


PyTorch implementation of the electronegativity equilibration (EEQ) model for atomic partial charges.
This module allows to process a single structure or a batch of structures for the calculation of atom-resolved dispersion energies.

For details on the EEQ model, see

- \S. A. Ghasemi, A. Hofstetter, S. Saha, and S. Goedecker, *Phys. Rev. B*, **2015**, *92*, 045131. DOI: [10.1103/PhysRevB.92.045131](https://doi.org/10.1103/PhysRevB.92.045131)

- \E. Caldeweyher, S. Ehlert, A. Hansen, H. Neugebauer, S. Spicher, C. Bannwarth and S. Grimme, *J. Chem. Phys.*, **2019**, *150*, 154122. DOI: [10.1063/1.5090222](https://dx.doi.org/10.1063/1.5090222)


For alternative implementations, also check out

- [multicharge](https://github.com/grimme-lab/multicharge): Implementation of the EEQ model in Fortran.

## Installation

### pip

_tad-multicharge_ can easily be installed with `pip`.

```sh
pip install tad-multicharge
```

### conda

_tad-multicharge_ is also available from `conda`.

```sh
conda install tad-multicharge
```

### From source

This project is hosted on GitHub at `tad-mctc/tad-multicharge <https://github.com/tad-mctc/tad-multicharge>`__.
Obtain the source by cloning the repository with

```sh
git clone https://github.com/tad-mctc/tad-multicharge
cd tad-multicharge
```

We recommend using a [conda](https://conda.io/) environment to install the package.
You can setup the environment manager using a [mambaforge](https://github.com/conda-forge/miniforge) installer.
Install the required dependencies from the conda-forge channel.

```sh
mamba env create -n torch -f environment.yaml
mamba activate torch
```

Install this project with `pip` in the environment

```sh
pip install .
```

The following dependencies are required

- [numpy](https://numpy.org/)
- [tad-mctc](https://github.com/tad-mctc/tad-mctc)
- [pytest](https://docs.pytest.org/) (tests only)
- [torch](https://pytorch.org/)


## Compatibility

| PyTorch \ Python | 3.8                | 3.9                | 3.10               | 3.11               | 3.12               |
|------------------|--------------------|--------------------|--------------------|--------------------|--------------------|
| 1.11.0           | :white_check_mark: | :white_check_mark: | :x:                | :x:                | :x:                |
| 1.12.1           | :white_check_mark: | :white_check_mark: | :white_check_mark: | :x:                | :x:                |
| 1.13.1           | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :x:                |
| 2.0.1            | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :x:                |
| 2.1.2            | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :x:                |
| 2.2.2            | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| 2.3.1            | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| 2.4.1            | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| 2.5.1            | :x:                | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |

Note that only the latest bug fix version is listed, but all preceding bug fix minor versions are supported.
For example, although only version 2.2.2 is listed, version 2.2.0 and 2.2.1 are also supported.

On macOS and Windows, PyTorch<2.0.0 does only support Python<3.11.


## Development

For development, additionally install the following tools in your environment.

```sh
mamba install black covdefaults mypy pre-commit pylint pytest pytest-cov pytest-xdist tox
pip install pytest-random-order
```

With pip, add the option `-e` for installing in development mode, and add `[dev]` for the development dependencies

```sh
pip install -e .[dev]
```

The pre-commit hooks are initialized by running the following command in the root of the repository.

```sh
pre-commit install
```

For testing all Python environments, simply run `tox`.

```sh
tox
```

Note that this randomizes the order of tests but skips "large" tests. To modify this behavior, `tox` has to skip the optional _posargs_.

```sh
tox -- test
```

## Examples

The following example shows how to calculate the EEQ partial charges and the corresponding electrostatic energy for a single structure.

```python
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
```

The next example shows the calculation of the electrostatic energy with a simpler API for a batch of structures.

```python
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
```

## Contributing

This is a volunteer open source projects and contributions are always welcome.
Please, take a moment to read the [contributing guidelines](CONTRIBUTING.md).

## License

This project is licensed under the Apache License, Version 2.0 (the "License"); you may not use this project's files except in compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
