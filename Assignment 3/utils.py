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
			perm = np.random.permutation(np.arange(self.data.shape[0]))
			self.data, self.labels = self.data[perm], self.labels[perm]

	def nextBatch(self, random_flip=False, random_crop=False):
		pos, batch_size = self.pos, self.batch_size
		if (pos + batch_size >= self.data.shape[0]):
			self.done_epoch = True
			self.pos = self.data.shape[0]
			batch_xs, batch_ys = self.data[pos : ], self.labels[pos : ]
		else:
			self.pos = pos + batch_size
			batch_xs, batch_ys = self.data[pos : pos + batch_size], self.labels[pos : pos + batch_size]

		if (random_flip):
			# random horizontal flip
			rnd = np.random.random(size=(batch_xs.shape[0], ))
			for i in range(batch_xs.shape[0]):
				if (rnd[i] > 0.5):
					batch_xs[i] = np.flip(batch_xs[i], axis=1)

		if (random_crop):
			# random crop after padding
			rnd = np.random.random(size=(batch_xs.shape[0], ))
			for i in range(batch_xs.shape[0]):
				if (rnd[i] > 0.5):
					batch_xs[i] = self.randomCrop(batch_xs[i], 4) 	# 4 - hard-coded

		return np.expand_dims(batch_xs, axis=1), batch_ys

	# Pads image by given pad and does random crop back to original size - assumed (N, H, W, C) format
	def randomCrop(self, image, pad):
		padded_image = np.pad(image, [(pad, pad), (pad, pad)], 'constant')
		r = np.random.random_integers(0, 2 * pad, size=(2, ))
		padded_image = padded_image[r[0] : r[0] + image.shape[0], r[1] : r[1] + image.shape[1]]
		return padded_image

	def doneEpoch(self):
		return self.done_epoch

def getAccuracy(model, data, labels, batch_size, use_gpu):
	data_loader = DataLoader(data, labels, batch_size)
	acc = 0.0
	while (not data_loader.doneEpoch()):
		batch_xs, batch_ys = data_loader.nextBatch()
		batch_xs, batch_ys = torch.Tensor(batch_xs), torch.Tensor(batch_ys)
		if (use_gpu):
			batch_xs, batch_ys = batch_xs.cuda(), batch_ys.cuda()
		scores = model.forward(batch_xs)
		acc += torch.sum(torch.argmax(scores, dim=1).long() == batch_ys.long()).item()

	acc = acc * 1.0 / data.shape[0]
	return acc

def getPredictions(model, data, batch_size, use_gpu):
	labels = np.zeros(data.shape[0])
	data_loader = DataLoader(data, labels, batch_size)
	pred = np.zeros(data.shape[0])
	pos = 0
	while (not data_loader.doneEpoch()):
		batch_xs, batch_ys = data_loader.nextBatch()
		batch_xs, batch_ys = torch.Tensor(batch_xs), torch.Tensor(batch_ys)
		if (use_gpu):
			batch_xs, batch_ys = batch_xs.cuda(), batch_ys.cuda()
		scores = model.forward(batch_xs)
		pred[pos : pos + batch_size] = torch.argmax(scores, dim=1).long().cpu().numpy()
		pos += batch_size
	
	return pred