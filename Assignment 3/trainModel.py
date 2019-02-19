import argparse
import sys
from Model import Model
from Linear import Linear
from Criterion import Criterion
from optim import MomentumOptimizer
from ReLU import ReLU
import torch
import numpy as np
import torchfile, pickle, os, sys
import utils
import math

torch.set_default_dtype(torch.double)

parser = argparse.ArgumentParser()
parser.add_argument('-modelName', help='name of model; name used to create folder to save model')
parser.add_argument('-data', help='path to train data')
parser.add_argument('-target', help='path to target labels')
args = parser.parse_args()

input = torchfile.load(args.data)
input = np.reshape(input, (input.shape[0], -1))
target = torchfile.load(args.target)
input = input.astype(np.float32)
min_val, max_val = 0.0, 255.0
input = (input - min_val) / (max_val - min_val) - 0.5

fraction_validation = 0.05

input_size = input.shape[1 : ]
output_size = (np.max(target) - np.min(target) + 1, )

np.random.seed(0)
torch.manual_seed(0)

data_perm = np.random.permutation(np.arange(input.shape[0]))
input, target = input[data_perm], target[data_perm]
tr_data, tr_labels, val_data, val_labels = utils.splitTrainVal(input, target, fraction_validation)

epochs = 20
lr = 0.01
reg = 0.0
momentum = 0.9
print_every = 100
batch_size = 128

tr_loader = utils.DataLoader(tr_data, tr_labels, batch_size)

model = Model()

model.addLayer(Linear(input_size[0], 200))
model.addLayer(ReLU())
model.addLayer(Linear(200, 100))
model.addLayer(ReLU())
model.addLayer(Linear(100, output_size[0]))

criterion = Criterion()

optim = MomentumOptimizer(model, lr=lr, reg=reg, momentum=momentum)

val_accs = []
loss = []
acc = []
i = 0
for epoch in range(epochs):
	tr_loader.resetPos()
	if (fraction_validation != 0.0):
		val_acc = utils.getAccuracy(model, val_data, val_labels, batch_size)
		val_accs.append(val_acc)
		print("Epoch : %d, validation accuracy : %f" % (epoch, val_acc))
	while (not tr_loader.doneEpoch()):
		batch_xs, batch_ys = tr_loader.nextBatch()
		batch_xs, batch_ys = torch.Tensor(batch_xs), torch.Tensor(batch_ys)
		scores = model.forward(batch_xs)
		cur_loss = criterion.forward(scores, batch_ys).item()
		cur_acc = torch.sum(torch.argmax(scores, dim=1).long() == batch_ys.long()).item() * 1.0 / batch_xs.shape[0]
		loss.append(cur_loss)
		acc.append(cur_acc) 
		d_scores = criterion.backward(scores, batch_ys)
		model.backward(batch_xs, d_scores)
		optim.step()
		if (i % print_every == 0):
			print("Train loss : %f, Train acc : %f" % (loss[-1], acc[-1]))

		i += 1

if (not os.path.exists(args.modelName)):
	os.makedirs(args.modelName)
with open(os.path.join(args.modelName, 'model.pt'), 'wb') as f:
	pickle.dump((model, criterion), f)