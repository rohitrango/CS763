import Linear, ReLU, Conv, Flatten, MaxPool
import torch
import sys

class MomentumOptimizer:
	def __init__(self, model, lr=0.1, reg=0.1, momentum=0.9):
		self.model = model
		self.lr = lr
		self.reg = reg
		self.momentum = momentum

		self.v = []
		for layer in self.model.Layers:
			if ((type(layer) == Linear.Linear) or (type(layer) == Conv.Conv)):
				self.v.append({'W' : torch.zeros_like(layer.W, device=layer.W.device), 'B' : torch.zeros_like(layer.B, device=layer.B.device)})
			elif ((type(layer) == ReLU.ReLU) or (type(layer) == Flatten.Flatten) or (type(layer) == MaxPool.MaxPool)):
				self.v.append({})
			else:
				raise NotImplementedError
				sys.exit(0)


	def step(self):
		for i in range(len(self.model.Layers)):
			if ((type(self.model.Layers[i]) == Linear.Linear) or (type(self.model.Layers[i]) == Conv.Conv)):
				self.v[i]['W'] = self.momentum * self.v[i]['W'] + (1 - self.momentum) * self.lr * (self.model.Layers[i].gradW + self.reg * self.model.Layers[i].W)
				self.v[i]['B'] = self.momentum * self.v[i]['B'] + (1 - self.momentum) * self.lr * (self.model.Layers[i].gradB + self.reg * self.model.Layers[i].B)
				
				self.model.Layers[i].W = self.model.Layers[i].W - self.v[i]['W']
				self.model.Layers[i].B = self.model.Layers[i].B - self.v[i]['B']

