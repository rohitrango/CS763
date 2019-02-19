import torch

torch.set_default_dtype(torch.double)

# Can't use nn package, so how do we create a torch class ? For now, a normal python class.
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
			Assuming input is (batch_size,num_classes)
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
			P.S. The gradients are again only calculated here, they need to be added too !
			Need to describe a step() sort of function to add the grads to the weights and biases
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

	def dispGradParam(self):
		for layer in reversed(self.Layers):
			layer.dispParam()

	def clearGradParam(self):
		for layer in self.Layers:
			layer.clearGrad()
			self.gradOutputs = None

	def addLayer(self, layer):
		self.Layers.append(layer)
