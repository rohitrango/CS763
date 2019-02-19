import torch
import math

# Can't use nn package, so how do we create a torch class ? For now, a normal python class.
class Flatten:

	def __init__(self):
		pass

	def forward(self, input):
		B = input.shape[0]
		output = input.reshape(B, -1)
		return output

	def backward(self, input, gradOutput):
		# gradoutput is a flattened guy
		gradInput = gradOutput.reshape(input.shape)
		return gradInput

	def clearGrad(self):
		pass

	def dispParam(self):
		pass
