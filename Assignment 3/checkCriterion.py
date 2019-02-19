import argparse
import sys
from Model import Model
from Linear import Linear
from ReLU import ReLU
import torch
import torchfile

torch.set_default_dtype(torch.double)

parser = argparse.ArgumentParser()
parser.add_argument('-i', help='path to input.bin')
parser.add_argument('-t', help='path to target.bin')
parser.add_argument('-ig' help='path to gradInput.bin')
args = parser.parse_args()

input = torch.Tensor(torchfile.load(args.i))
target = torch.Tensor(torchfile.load(args.t))

criterion = Criterion()
loss = criterion.forward(input, target)
print(loss)
gradInput = criterion.backward(input, target)

# INCOMPLETE
# save gradInput

