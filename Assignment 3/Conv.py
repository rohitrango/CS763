import torch
import math

torch.set_default_dtype(torch.double)

# Can't use nn package, so how do we create a torch class ? For now, a normal python class.
class Conv:

	def __init__(self, channels_in, channels_out, H, W, stride=1):
		'''
			Random initialization of the weights, the gradients are calculated anyway, so initialization doesn't matter
		'''
		self.channels_in = channels_in
		self.channels_out = channels_out
		self.h = H
		self.w = W
		self.stride = stride

		# W and b are of some output sizes
		self.W = torch.randn(channels_out, channels_in, H, W)
		self.W = self.W * math.pow(2 / (channels_in*H*W), 0.5) 			# CORR: xavier initialization: not sure if it is useful when using ReLU
		self.B = torch.zeros(channels_out) + 0.01
		self.gradW = torch.zeros_like(self.W)
		self.gradB = torch.zeros_like(self.B)

	def forward(self, input):
		'''
			Assuming input is (N, C, H, W) as output is required to be (N, F, H1, W1)
			Conv weights are of size (F, C, H_kernel, W_kernel)
		'''
		N, C, H, W = input.shape
		H_out = (H - self.h)//self.stride + 1
		W_out = (W - self.w)//self.stride + 1

		## H and W
		flatW = self.W.reshape(self.W.shape[0], -1)
		output = torch.zeros((N, self.channels_out, H_out, W_out))

		ii_idx = list(range(H_out))
		jj_idx = list(range(W_out))
		for ii in ii_idx:
			for jj in jj_idx:
				# chunk = N * C * k1 * k2, W = F * C * k1 * k2
				inp_ii = ii*self.stride
				inp_jj = jj*self.stride
				# Compute conv
				chunk = input[:, :, inp_ii:inp_ii+self.h, inp_jj:inp_jj+self.w]
				flatchunk = chunk.reshape(chunk.shape[0], -1)
				outchunk = torch.mm(flatchunk, flatW.t())
				output[:, :, ii, jj] = outchunk + self.B[None]

		return output


	def backward(self, input, gradOutput):
		'''
		input is of size N * C * H * W
		gradoutput = N * F * H1 * W1
		'''
		N, C, H, W = input.shape
		H_out = (H - self.h)//self.stride + 1
		W_out = (W - self.w)//self.stride + 1

		gradInput = torch.zeros_like(input)

		self.gradB = gradOutput.sum(dim=[0, 2, 3])
		flatW = self.W.reshape(self.W.shape[0], -1)
		## Find gradients
		ii_idx = list(range(H_out))
		jj_idx = list(range(W_out))
		for ii in ii_idx:
			for jj in jj_idx:
				# Take gradients from output pixel and put in into weights and bias
				inp_ii = ii*self.stride
				inp_jj = jj*self.stride

				# N * C * k1 * k2
				chunk = input[:, :, inp_ii:inp_ii+self.h, inp_jj:inp_jj+self.w]
				# N * F 
				gradOutChunk = gradOutput[:, :, ii, jj]
				dW = torch.mm(gradOutChunk.t(), chunk.reshape(chunk.shape[0], -1))
				dW = dW.reshape(gradOutChunk.shape[1], *chunk.shape[1:])
				self.gradW = self.gradW + dW

				#### Get gradient for input chunk here
				#  N * F ------- F * C * k1 * k2
				dx = torch.mm(gradOutChunk, flatW).reshape(self.channels_out, C, self.h, self.w)
				gradInput[:, :, inp_ii:inp_ii+self.h, inp_jj:inp_jj+self.w] = gradInput[:, :, inp_ii:inp_ii+self.h, inp_jj:inp_jj+self.w] + dx

		return gradInput


	def clearGrad(self):
		self.gradW = torch.zeros_like(self.W)
		self.gradB = torch.zeros_like(self.B)

	def dispParam(self):
		pass



if __name__ == '__main__':
	convOurs = Conv(3, 32, 5, 5, stride=3)
	convNorm = torch.nn.Conv2d(3, 32, 5, stride=3)
	convNorm.weight.data = convOurs.W.data
	convNorm.bias.data = convOurs.B.data.squeeze()

	inp = torch.rand(32, 3, 28, 28)
	out1 = convOurs.forward(inp)
	out2 = convNorm.forward(inp)
	# Check values in forward prop
	print(torch.max(torch.abs(out1 - out2)))

	### Get a loss function
	s = out2.sum()
	s.backward()

	gradOutput = torch.ones(out2.shape)
	gradInp = convOurs.backward(inp, gradOutput)

	# Backward pass comparison
	print(torch.max(torch.abs(convOurs.gradW - convNorm.weight.grad)))
	print(torch.max(torch.abs(convOurs.gradB - convNorm.bias.grad)))
