import argparse
import sys, os
from Model import Model
from Linear import Linear
from ReLU import ReLU
import torch
import numpy as np
import torchfile


torch.set_default_dtype(torch.double)

parser = argparse.ArgumentParser()
parser.add_argument('-config', help='path to modelConfig.txt')
parser.add_argument('-i', help='path to input.bin')
parser.add_argument('-og', help='path to gradOutput.bin')
parser.add_argument('-o', help='path to output.bin')
parser.add_argument('-ow', help='path to gradWeight.bin')
parser.add_argument('-ob', help='path to gradB.bin')
parser.add_argument('-ig', help='path to gradInput.bin')
args = parser.parse_args()

model = Model()

with open(args.config, 'r') as f:
	num_layers = int(f.readline())
	for _ in range(num_layers):
		line = f.readline().split()
		if (line[0] == 'linear'):
			model.addLayer(Linear(int(line[1]), int(line[2])))
		elif (line[0] == 'relu'):
			model.addLayer(ReLU())
		else:
			print("not implemented")
			sys.exit(0)

	line = f.readline().split()[0]
	if (line[0][-1] == '\n'):
		W = torchfile.load(os.path.join(line[: -1]))
	else:
		W = torchfile.load(os.path.join(line))
	line = f.readline().split()[0]
	if (line[0][-1] == '\n'):
		B = torchfile.load(os.path.join(line[:-1]))
	else:
		B = torchfile.load(os.path.join(line))

j = 0
for i in range(num_layers):
	if (type(model.Layers[i]) == Linear):
		model.Layers[i].W = torch.Tensor(W[j])
		model.Layers[i].B = torch.Tensor(np.expand_dims(B[j], axis=1))
		j += 1

input = torchfile.load(args.i)
input = np.reshape(input, (input.shape[0], -1))
input = torch.Tensor(input)
model.clearGradParam()
model.forward(input)
gradOutput = torch.Tensor(torchfile.load(args.og))
model.backward(input, torch.Tensor(gradOutput))
gradW = []
gradB = []

for i in range(num_layers):
	if (type(model.Layers[i]) == Linear):
		gradW.append(model.Layers[i].gradW.numpy())
		gradB.append(np.squeeze(model.Layers[i].gradB.numpy(), axis=1))

gradInput = model.gradOutputs[0].numpy()
gradInput = np.reshape(gradInput, input.shape)

# gradW_t = torchfile.load(args.ow)
# gradB_t = torchfile.load(args.ob)
# gradInput_t = torchfile.load(args.ig)

# mse_W = 0.0
# for i in range(len(gradW)):
# 	mse_W += np.mean((gradW[i] - gradW_t[i]) ** 2)

# mse_W /= len(gradW)

# mse_B = 0.0
# for i in range(len(gradB)):
# 	mse_B += np.mean((gradB[i] - gradB_t[i]) ** 2)

# mse_B /= len(gradB)

# mse_gradInput = 0.0
# for i in rage(len(gradInput)):
# 	mse_gradInput += np.mean((gradInput[i] - gradInput_t[i]) ** 2)

# mse_gradInput /= len(gradInput)

# print("mse_W:", mse_W)
# print("mse_B:", mse_B)
# print("mse_gradInput:", mse_gradInput)

torch.save(gradW, args.ow)
torch.save(gradB, args.ob)
torch.save(gradInput, args.ig)