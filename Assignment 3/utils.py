import numpy as np
import torch

def splitTrainVal(input, target, fraction_validation=0.0):
	if (fraction_validation == 0.0):
		return input, target, None, None
	else:
		val_size = int(input.shape[0] * fraction_validation)
		train_size = input.shape[0] - val_size
		if (val_size == 0):
			return input, target, None, None
		else:
			return input[0 : train_size], target[0 : train_size], input[train_size : ], target[train_size : ]

class DataLoader:
	def __init__(self, data, labels, batch_size):
		self.data = data
		self.labels = labels
		self.pos = 0
		self.batch_size = batch_size
		self.done_epoch = False

	def resetPos(self, shuffle=False):
		self.pos = 0
		self.done_epoch = False
		if (shuffle):
			perm = np.random.permutation(np.arage(self.data.shape[0]))
			self.data, self.labels = self.data[perm], self.labels[perm]

	def nextBatch(self):
		pos, batch_size = self.pos, self.batch_size
		if (pos + batch_size >= self.data.shape[0]):
			self.done_epoch = True
			self.pos = self.data.shape[0]
			return self.data[pos : ], self.labels[pos : ]
		else:
			self.pos = pos + batch_size
			return self.data[pos : pos + batch_size], self.labels[pos : pos + batch_size]

	def doneEpoch(self):
		return self.done_epoch

def getAccuracy(model, data, labels, batch_size):
	data_loader = DataLoader(data, labels, batch_size)
	acc = 0.0
	while (not data_loader.doneEpoch()):
		batch_xs, batch_ys = data_loader.nextBatch()
		batch_xs, batch_ys = torch.Tensor(batch_xs), torch.Tensor(batch_ys)
		scores = model.forward(batch_xs)
		acc += torch.sum(torch.argmax(scores, dim=1).long() == batch_ys.long()).item()

	acc = acc * 1.0 / data.shape[0]
	return acc