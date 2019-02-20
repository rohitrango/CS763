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
		self.W = torch.rand(channels_out, channels_in, H, W)
		rand_bound = math.pow(6 / (channels_in*H*W), 0.5)
		self.W = 2 * rand_bound * self.W - rand_bound 			# CORR: xavier initialization: not sure if it is useful when using ReLU
		rand_bound = math.pow(1 / (channels_in*H*W), 0.5)
		self.B = 2 * rand_bound * torch.rand(channels_out) - rand_bound
		self.gradW = torch.zeros_like(self.W)
		self.gradB = torch.zeros_like(self.B)

	def cuda(self):
		# Utilize GPU memory (we have GPUs, don't ask)
		self.W = self.W.cuda()
		self.B = self.B.cuda()
		self.gradW = self.gradW.cuda()
		self.gradB = self.gradB.cuda()
		return self


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
		output = torch.zeros((N, self.channels_out, H_out, W_out), device=input.device)

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

		gradInput = torch.zeros_like(input, device=input.device)

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
				dx = torch.mm(gradOutChunk, flatW).reshape(-1, C, self.h, self.w)
				gradInput[:, :, inp_ii:inp_ii+self.h, inp_jj:inp_jj+self.w] = gradInput[:, :, inp_ii:inp_ii+self.h, inp_jj:inp_jj+self.w] + dx

		return gradInput


	def clearGrad(self):
		self.gradW = torch.zeros_like(self.W, device=self.W)
		self.gradB = torch.zeros_like(self.B, device=self.B)

	def dispParam(self):
		pass



if __name__ == '__main__':
	convOurs = Conv(3, 32, 5, 5, stride=3)
	convOurs.cuda()
	convNorm = torch.nn.Conv2d(3, 32, 5, stride=3).cuda()
	convNorm.weight.data = convOurs.W.data
	convNorm.bias.data = convOurs.B.data.squeeze()

	inp = torch.rand(32, 3, 28, 28).cuda()
	out1 = convOurs.forward(inp)
	out2 = convNorm.forward(inp)
	# Check values in forward prop
	print(torch.max(torch.abs(out1 - out2)))

	### Get a loss function
	s = out2.sum()
	s.backward()

	gradOutput = torch.ones(out2.shape).cuda()
	gradInp = convOurs.backward(inp, gradOutput)

	# Backward pass comparison
	print(torch.max(torch.abs(convOurs.gradW - convNorm.weight.grad)))
	print(torch.max(torch.abs(convOurs.gradB - convNorm.bias.grad)))
	raw_input()
