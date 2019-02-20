import sys
sys.path.append('../')

import torch
import torch.nn as nn
import numpy as np
import argparse, os, torchfile
import models
import utils
import pickle
import matplotlib.pyplot as plt
import time

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
parser.add_argument('--use_dropout', action='store_true', help='whether to use dropout')
parser.add_argument('--random_flip', action='store_true', help='whether to use random flip data augmentation')
parser.add_argument('--random_crop', action='store_true', help='whether to use random crop data augmentation')
parser.add_argument('--use_gpu', action='store_true', help='whether to use gpu for training/testing')

args = parser.parse_args()

input = torchfile.load(args.data)
target = torchfile.load(args.target)
input = input.astype(np.float32)
min_val, max_val = 0.0, 255.0
input = (input - min_val) / (max_val - min_val) - 0.5

fraction_validation = args.fraction_validation

input_shape = input.shape[1 : ]
output_shape = (np.max(target) - np.min(target) + 1, )

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

if (args.use_dropout):
	dropout = (0.2, 0.5)
else:
	dropout = (0.0, 0.0)

model = models.BNConvNetworkSmall1NoPadding(input_shape, output_shape)

cpu_device = torch.device('cpu')
if (args.use_gpu):
	fast_device = torch.device('cuda:0')
else:
	fast_device = cpu_device

model = model.to(fast_device)

criterion = nn.CrossEntropyLoss()
optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=reg)
lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lambda epoch : 0.1 ** (epoch // 60))
val_accs = []
loss = []
acc = []
i = 0

if (not os.path.exists(args.modelName)):
	os.makedirs(args.modelName)

def getAccuracy(model, data, labels, batch_size, fast_device):
	data_loader = utils.DataLoader(data, labels, batch_size)
	acc = 0.0
	while (not data_loader.doneEpoch()):
		batch_xs, batch_ys = data_loader.nextBatch()
		batch_xs, batch_ys = torch.Tensor(batch_xs).to(fast_device), torch.Tensor(batch_ys).to(fast_device).long()
		scores = model(batch_xs)
		acc += torch.sum(torch.argmax(scores, dim=1).long() == batch_ys.long()).item()

	acc = acc * 1.0 / data.shape[0]
	return acc

start_time = time.time()
for epoch in range(epochs):
	tr_loader.resetPos()
	if (fraction_validation != 0.0):
		model = model.eval()
		val_acc = getAccuracy(model, val_data, val_labels, batch_size, fast_device)
		val_accs.append(val_acc)
		print("Epoch : %d, validation accuracy : %f" % (epoch, val_acc))
		print("Time Elapsed:", time.time() - start_time)
	while (not tr_loader.doneEpoch()):
		model = model.train()
		batch_xs, batch_ys = tr_loader.nextBatch(random_flip=args.random_flip, random_crop=args.random_crop)
		batch_xs, batch_ys = torch.Tensor(batch_xs).to(fast_device), torch.Tensor(batch_ys).to(fast_device).long()
		optim.zero_grad()
		scores = model(batch_xs)
		cur_loss = criterion(scores, batch_ys)
		cur_acc = torch.sum(torch.argmax(scores, dim=1).long() == batch_ys.long()).item() * 1.0 / batch_xs.shape[0]
		loss.append(cur_loss)
		acc.append(cur_acc) 
		cur_loss.backward()
		optim.step()
		if (i % print_every == 0):
			print("Train loss : %f, Train acc : %f" % (loss[-1], acc[-1]))
		i += 1

	if (epoch % save_every == 0):
		torch.save({'model' : model	, 
					'criterion' : criterion}, os.path.join(args.modelName, 'model_' + str(epoch) + '.pt'))

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