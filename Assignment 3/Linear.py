import torch
import numpy as np

torch.set_default_dtype(torch.double)

# Can't use nn package, so how do we create a torch class ? For now, a normal python class.
class Linear():

	def __init__(self, num_in, num_out):
		'''
			Random initialization of the weights, the gradients are calculated anyway, so initialization doesn't matter
		'''
		self.W = torch.randn(num_out,num_in)
		self.B = torch.randn(num_out,1)
		self.gradW = torch.zeros_like(self.W)
		self.gradB = torch.zeros_like(self.B)

	def forward(self, input):
		'''
			Assuming input is (batch_size,num_in) as output is required to be (batch_size, num_out)
		'''
		self.output = torch.matmul(input,torch.t(self.W))
		self.output = self.output + torch.t(self.B)
		output = self.output + 0
		return output

	def backward(self, input, gradOutput):
		'''
			gradInput is (batch_size,num_in) and gradOutput is similar
			Not taking into account the activation function here, obviously
			Derivative only of the unactivated output wrt the weights
			Also, not updating the weights here, just calculating the gradients
			Finally, the gradients of W,B are being added up for all batch_examples
			Not yet normalised (Should ideally be!) as PS wants us to do it separately as BatchNorm layer class.
		'''
		self.gradW = torch.t(torch.matmul(torch.t(input),self.output))
		self.gradB = torch.t(torch.sum(gradOutput,0))
		self.gradInput = torch.matmul(gradOutput,self.W)
		gradInput = self.gradInput + 0
		return gradInput