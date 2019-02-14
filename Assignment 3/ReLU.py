import torch
import numpy as np

torch.set_default_dtype(torch.double)

# Can't use nn package, so how do we create a torch class ? For now, a normal python class.
class ReLU():

	def __init__(self):
		'''
			Empty function for now
		'''
		pass

	def forward(self, input):
		'''
			Assuming input is (batch_size,num_in) as output is required to be (batch_size, num_out)
		'''
		self.output = torch.max(input,torch.zeros_like(input))
		output = self.output + 0
		return output

	def backward(self, input, gradOutput):
		'''
			gradInput is (batch_size,num_in) and gradOutput is similar
			This is the derivative wrt activation function, taking grad at x = 0 as 0
		'''
		self.gradInput = torch.max(gradOutput,torch.zeros_like(gradOutput))
		self.gradInput[self.gradInput > 0] = 1
		self.gradInput = self.gradInput * gradOutput
		gradInput = self.gradInput + 0
		return gradInput