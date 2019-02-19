import argparse
import sys
from Model import Model
from Linear import Linear
from Criterion import Criterion
from optim import MomentumOptimizer
from ReLU import ReLU
from Conv import Conv
from Flatten import Flatten
from MaxPool import MaxPool
import torch
import numpy as np
import torchfile, pickle, os, sys
import utils
import math
import matplotlib.pyplot as plt 						# CHECK : finally remove this package

torch.set_default_dtype(torch.double)

parser = argparse.ArgumentParser()
parser.add_argument('-modelName', help='name of model; name used to create folder to save model')
parser.add_argument('-data', help='path to train data')
parser.add_argument('-target', help='path to target labels')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--epochs', type=int, default=50, help='number of epochs')
parser.add_argument('--reg', type=float, default=0.0, help='regularization weight')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum in momentum optimizer')
parser.add_argument('--batch_size', type=int, default=128, help='batch size for training, testing')
parser.add_argument('--fraction_validation', type=float, default=0.1, help='fraction of data to be used for validation')
parser.add_argument('--use_gpu', action='store_true', help='whether to use gpu for training/testing')

args = parser.parse_args()

input = torchfile.load(args.data)
target = torchfile.load(args.target)
input = input.astype(np.float32)
min_val, max_val = 0.0, 255.0
input = (input - min_val) / (max_val - min_val) - 0.5

fraction_validation = args.fraction_validation

input_size = input.shape[1 : ]
output_size = (np.max(target) - np.min(target) + 1, )

np.random.seed(0)
torch.manual_seed(0)

data_perm = np.random.permutation(np.arange(input.shape[0]))
input, target = input[data_perm], target[data_perm]
tr_data, tr_labels, val_data, val_labels = utils.splitTrainVal(input, target, fraction_validation)

epochs = args.epochs
lr = args.lr
reg = args.reg
momentum = args.momentum
print_every = 100
save_every = 20
batch_size = args.batch_size

tr_loader = utils.DataLoader(tr_data, tr_labels, batch_size)

model = Model()

model.addLayer(Conv(1, 16, 3, 3))
model.addLayer(ReLU())
model.addLayer(MaxPool(2))
model.addLayer(Conv(16, 16, 3, 3))
model.addLayer(ReLU())
model.addLayer(MaxPool(2))
model.addLayer(Conv(16, 16, 3, 3))
model.addLayer(ReLU())
model.addLayer(MaxPool(6))

model.addLayer(Flatten())

model.addLayer(Linear(16 * 3 * 3, 32))
# model.addLayer(Linear(108 * 108, 32))
model.addLayer(ReLU())
model.addLayer(Linear(32, output_size[0]))

model = model.cuda()

criterion = Criterion()

optim = MomentumOptimizer(model, lr=lr, reg=reg, momentum=momentum)

val_accs = []
loss = []
acc = []
i = 0
for epoch in range(epochs):
	tr_loader.resetPos()
	if (fraction_validation != 0.0):
		val_acc = utils.getAccuracy(model, val_data, val_labels, batch_size, args.use_gpu)
		val_accs.append(val_acc)
		print("Epoch : %d, validation accuracy : %f" % (epoch, val_acc))
		start_time = time.time()
	while (not tr_loader.doneEpoch()):
		batch_xs, batch_ys = tr_loader.nextBatch()
		batch_xs, batch_ys = torch.Tensor(batch_xs), torch.Tensor(batch_ys)
		if (args.use_gpu):
			batch_xs, batch_ys = batch_xs.cuda(), batch_ys.cuda()
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
	if (epoch % save_every == 0):
		torch.save({'model' : model	, 
					'criterion' : criterion}, os.path.join(args.modelName, 'model_' + str(epoch) + '.pt'))

if (not os.path.exists(args.modelName)):
	os.makedirs(args.modelName)

torch.save({'model' : model	, 
			'criterion' : criterion}, os.path.join(args.modelName, 'model_final.pt'))

# CHECK : finally remove the part below this
with open(os.path.join(args.modelName, 'stats.bin'), 'wb') as f:
	pickle.dump((val_accs, loss, acc), f)

with open(os.path.join(args.modelName, 'stats.txt'), 'w') as f:
	f.write('Validation accuracy : %f' % (val_accs[-1]))

plt.plot(val_accs)
plt.savefig(os.path.join(args.modelName, 'val_acc_graph.pdf'))
plt.clf()

plt.plot(loss)
plt.savefig(os.path.join(args.modelName, 'loss_graph.pdf'))
plt.clf()
plt.plot(acc)
plt.savefig(os.path.join(args.modelName, 'acc_graph.pdf'))
plt.clf()