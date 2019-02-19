import torch
import math

torch.set_default_dtype(torch.double)

# Can't use nn package, so how do we create a torch class ? For now, a normal python class.
class Linear:

	def __init__(self, num_in, num_out):
		'''
			Random initialization of the weights, the gradients are calculated anyway, so initialization doesn't matter
		'''
		self.W = torch.randn(num_out,num_in)
		self.W = self.W * math.pow(2 / (num_in + num_out), 0.5) 			# CORR: xavier initialization: not sure if it is useful when using ReLU
		self.B = torch.zeros(num_out,1)
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
		'''
		batch_size = input.size(0)
		self.gradW = torch.t(torch.matmul(torch.t(input),gradOutput))
		# self.gradW = self.gradW/batch_size 								# CORR: redundant; already done in loss function
		self.gradB = torch.t(torch.sum(gradOutput, dim=0).unsqueeze(0))
		# self.gradB = self.gradB/batch_size								# CORR: redundant; already done in loss function
		gradInput = torch.matmul(gradOutput,self.W)
		return gradInput

	def clearGrad(self):
		self.gradW = torch.zeros_like(self.W)
		self.gradB = torch.zeros_like(self.B)

	def dispParam(self):
		'''
			Display parameters in 2D Matrix format with elements separated by spaces
		'''
		weight = self.W
		bias = self.B
		print('W')
		for i in range(weight.size(0)):
			for j in range(weight[i].size(0)):
				print(weight[i][j].item(),end=' ')
			print()

		print('b')
		for i in range(bias.size(0)):
			print(bias[i].item(),end=' ')
		print()
		
		print()