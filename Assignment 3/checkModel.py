import argparse
import sys
from Model import Model
from Linear import Linear
from ReLU import ReLU
import torch
import torchfile


torch.set_default_dtype(torch.double)

parser = argparse.ArgumentParser()
parser.add_argument('-config', help='path to modelConfig.txt')
parser.add_argument('-i', help='path to input.bin')
parser.add_argument('--og', help='path to gradOutput.bin')
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

	line = f.readline()
	W = torchfile.load(line)
	line = f.readline()
	B = torchfile.load(line)

j = 0
for i in range(num_layers):
	if (type(model.Layers[i]) == Linear):
		model.Layers[i].W = torch.Tensor(W[j])
		model.Layers[i].B = torch.Tensor(B[j])
		j += 1

input = torchfile.load(args.i)
input = np.reshape(inputs, (inputs.shape[0], -1))
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
		gradB.append(model.Layers[i].gradB.numpy())

gradInput = model.gradOutputs[0].numpy()
gradInput = np.reshape(gradInput, input.shape)

# gradW_t = torchfile.load(args.ow)
# gradB_t = torchfile.load(args.ob)
# gradInput_t = torchfile.load(args.ig)

# mse_W = 0.0
# for i in range(len(gradW)):
# 	mse_W += np.mean((gradW - gradW_t) ** 2)

# mse_W /= len(gradW)

# mse_B = 0.0
# for i in range(len(gradB)):
# 	mse_B += np.mean((gradB - gradB_t) ** 2)

# mse_B /= len(gradB)

# mse_gradInput = 0.0
# for i in rage(len(gradInput)):
# 	mse_gradInput += np.mean((gradInput - gradInput_t) ** 2)

# mse_gradInput /= len(gradInput)

torch.save(gradW, args.ow)
torch.save(gradB, args.ob)
torch.save(gradInput, args.ig)