import RNN, MultiRNN
import torch
import sys

class SGD:
	def __init__(self, model, lr=0.1, reg=0.1, momentum=0.9):
		self.model = model
		self.lr = lr
		self.reg = reg
		self.momentum = momentum

		self.v = []
		for layer in self.model.Layers:
			self.v.append({})
			if (type(layer) == RNN.RNN or type(layer) == MultiRNN.MultiRNN):
				for k, v in layer.parameters.items():
					self.v[-1][k] = torch.zeros_like(layer.parameters[k], device=layer.parameters[k].device)
			else:
				raise NotImplementedError


	def step(self):
		for i in range(len(self.model.Layers)):
			layer = self.model.Layers[i]
			if (type(layer) == RNN.RNN or type(layer) == MultiRNN.MultiRNN):
				for k, _ in layer.parameters.items():
					self.v[i][k] = self.momentum * self.v[i][k] + (1 - self.momentum) * self.lr * (layer.gradients[k] + self.reg * layer.parameters[k])
					with torch.no_grad():
						# old_parameters = layer.parameters[k].detach().clone()
						# print("Value of parameters before update", layer.parameters[k])
						
						layer.parameters[k].copy_(layer.parameters[k] - self.v[i][k])
						
						# print("Current momentum", self.v[i][k])
						# print("Current gradient", layer.gradients[k] + self.reg * layer.parameters[k])
						# print("New value of parameters", layer.parameters[k])
						# if (torch.sum(old_parameters != layer.parameters[k]) == 0):
						# 	print('Panic!! Not getting updated')
						# input()


	def zero_grad(self):
		"""
		Initializes the gradient tensor to zero in the model
		"""
		self.model.zero_grad()

class Adam:
	def __init__(self, model, lr=0.1, reg=0.1, beta1=0.9, beta2=0.999, eps=1e-8):
		self.model = model
		self.lr = lr
		self.reg = reg
		self.beta1 = beta1
		self.beta2 = beta2
		self.eps = eps
		self.t = 0

		self.m = []
		self.v = []
		for layer in self.model.Layers:
			self.v.append({})
			self.m.append({})
			if (type(layer) == RNN.RNN or type(layer) == MultiRNN.MultiRNN):
				for k, v in layer.parameters.items():
					self.v[-1][k] = torch.zeros_like(layer.parameters[k], device=layer.parameters[k].device)
					self.m[-1][k] = torch.zeros_like(layer.parameters[k], device=layer.parameters[k].device)
			else:
				raise NotImplementedError


	def step(self):
		self.t += 1
		for i in range(len(self.model.Layers)):
			layer = self.model.Layers[i]
			if (type(layer) == RNN.RNN or type(layer) == MultiRNN.MultiRNN):
				for k, _ in layer.parameters.items():
					grad = layer.gradients[k] + self.reg * layer.parameters[k]
					self.m[i][k] = self.beta1 * self.m[i][k] + (1 - self.beta1) * grad
					self.v[i][k] = self.beta2 * self.v[i][k] + (1 - self.beta2) * torch.pow(grad, 2)
					m = self.m[i][k] / (1 - self.beta1 ** self.t)
					v = self.v[i][k] / (1 - self.beta2 ** self.t)
					with torch.no_grad():
						# old_parameters = layer.parameters[k].detach().clone()
						# print("Value of parameters before update", layer.parameters[k])
						
						layer.parameters[k].copy_(layer.parameters[k] - self.lr * (m / (torch.pow(v, 0.5) + self.eps)))
						
						# print("Current momentum", self.v[i][k])
						# print("Current gradient", layer.gradients[k] + self.reg * layer.parameters[k])
						# print("New value of parameters", layer.parameters[k])
						# if (torch.sum(old_parameters != layer.parameters[k]) == 0):
						# 	print('Panic!! Not getting updated')
						# input()


	def zero_grad(self):
		"""
		Initializes the gradient tensor to zero in the model
		"""
		self.model.zero_grad()
