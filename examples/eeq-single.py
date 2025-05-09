# SPDX-Identifier: CC0-1.0
import torch

from tad_multicharge.model import eeq

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
qat, energy = eeq_model.solve(
    numbers, positions, total_charge, cn, return_energy=True
)

print(torch.sum(energy, -1))
# tensor(-0.1750)
print(qat)
# tensor([-0.8347, -0.8347,  0.2731,  0.2886,  0.2731,  0.2731,  0.2886,  0.2731])
