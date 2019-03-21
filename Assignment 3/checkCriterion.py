import argparse
import sys
from Model import Model
from Linear import Linear
from ReLU import ReLU
from Criterion import Criterion
import torch
import numpy as np
import torchfile

torch.set_default_dtype(torch.double)

parser = argparse.ArgumentParser()
parser.add_argument('-i', help='path to input.bin')
parser.add_argument('-t', help='path to target.bin')
parser.add_argument('-ig', help='path to gradInput.bin')
args = parser.parse_args()

input = torch.Tensor(torchfile.load(args.i))
target = torch.Tensor(torchfile.load(args.t))

# print(input.shape)
# print(target.shape)
# print(input)
# print(target)

criterion = Criterion()
loss = criterion.forward(input, target)
print(loss.item())
gradInput = criterion.backward(input, target).numpy()

# input.requires_grad_(True)
# criterion_t = torch.nn.CrossEntropyLoss()
# loss = criterion_t(input, target.long())
# loss.backward()

# gradInput_t = torchfile.load(args.ig)

# print("MSE_gradInput:", np.mean((gradInput - input.grad.detach().numpy()) ** 2))

torch.save(gradInput, args.ig)

