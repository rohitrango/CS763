import torch

torch.set_default_dtype(torch.double)

class Model:

	def __init__(self, isTrain=True):
		'''
			self.isTrain is optional and may not be needed, but kept for now
		'''
		self.Layers = []
		self.isTrain = isTrain
		self.outputs = []

	def cuda(self):
		for i in range(len(self.Layers)):
			self.Layers[i] = self.Layers[i].cuda()

		return self

	def forward(self, input):
		'''
			Assuming input is (batch_size, seq_len, num_classes)
		'''
		output = input + 0
		self.outputs = []
		for layer in self.Layers:
			output = layer.forward(output)
			self.outputs.append(output)
		return output

	def backward(self, input, gradOutput):
		'''
		Performs gradient calculation through backpropagation.
		P.S. 
			We need to decide whether we add the gradients during the backward step 
			or make a separate step and only accumulate the gradients here
		'''
		self.gradOutputs = []
		gradOut = gradOutput
		self.gradOutputs.insert(0, gradOut)
		for i in range(len(self.Layers)-1,-1,-1):
			
			prev_layer = 0
			inp = 0

			if i-1 < 0:
				inp = input
			else:
				inp = self.outputs[i-1]

			gradOut = self.Layers[i].backward(inp, gradOut)
			self.gradOutputs.insert(0, gradOut)

	def addLayer(self, layer):
		self.Layers.append(layer)
