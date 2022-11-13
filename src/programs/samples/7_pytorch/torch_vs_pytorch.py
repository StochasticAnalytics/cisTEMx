#!/usr/bin/python3

import torch

# Create a tensor
x = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# Print each element of the tensor
for i in range(x.size()[0]):
    print(x[i])