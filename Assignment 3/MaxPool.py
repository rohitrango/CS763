import torch
import math

torch.set_default_dtype(torch.double)

# Can't use nn package, so how do we create a torch class ? For now, a normal python class.
class MaxPool:

	def __init__(self, kernel_size):
		'''
			Random initialization of the weights, the gradients are calculated anyway, so initialization doesn't matter
		'''
		self.kernel_size = kernel_size

	def cuda(self):
		return self

	def forward(self, input):
		'''
			Assuming input is (N, C, H, W) as output is required to be (N, F, H1, W1)
			Conv weights are of size (F, C, H_kernel, W_kernel)
		'''
		N, C, H, W = input.shape
		H_out = H//self.kernel_size
		W_out = W//self.kernel_size

		# Code here
		unfoldedInp = input.unfold(2, self.kernel_size, self.kernel_size).contiguous()
		unfoldedInp = unfoldedInp.unfold(3, self.kernel_size, self.kernel_size).contiguous()
		# unfolded is of the same shape as (b, c, outH, outW, k1, k2)
		maxVal = torch.max(torch.max(unfoldedInp, 4)[0], 4)[0]
		output = maxVal
		return output


	def backward(self, input, gradOutput):
		'''
		(N, C, H, W)
		'''
		gradInput = torch.zeros_like(input, device=input.device)

		N, C, H, W = input.shape
		H_out = H//self.kernel_size
		W_out = W//self.kernel_size

		# Code here
		unfoldedInp = input.unfold(2, self.kernel_size, self.kernel_size).contiguous()
		unfoldedInp = unfoldedInp.unfold(3, self.kernel_size, self.kernel_size).contiguous()
		# unfolded is of the same shape as (b, c, outH, outW, k1, k2)
		maxVal = torch.max(torch.max(unfoldedInp, 4)[0], 4)[0]
		# maxVal is of (b, c, outH, outW)
		maxValGates = (torch.abs(unfoldedInp - maxVal[:, :, :, :, None, None])<1e-8).double()
		# maxValGates is of size (b, c, outH, outW, k1, k2) containing only 1 max per num
		# gradOutput is of size (b, c, outH, outW)
		maxValGates = maxValGates*gradOutput[:, :, :, :, None, None]
		# collapse the maxVal gates back
		gradInput = maxValGates.permute(0, 1, 4, 5, 2, 3).contiguous()
		# (N, C, h, w, H, W)
		gradInput = gradInput.reshape(N, C*self.kernel_size*self.kernel_size, H_out*W_out)
		gradInput = torch.nn.functional.fold(gradInput, (H, W), (self.kernel_size, self.kernel_size), stride=self.kernel_size)

		# Naive maxpool
		# for ii in range(H_out):
		# 	for jj in range(W_out):
		# 		inp_ii = self.kernel_size*ii
		# 		inp_jj = self.kernel_size*jj
		# 		chunk = input[:, :, inp_ii:inp_ii+self.kernel_size, inp_jj:inp_jj+self.kernel_size]
		# 		# chunk is of size N * C * K * K
		# 		maxVal = torch.max(torch.max(chunk, 2)[0], 2)[0][:, :, None, None]
		# 		gradInput[:, :, inp_ii:inp_ii+self.kernel_size, inp_jj:inp_jj+self.kernel_size] =\
		# 				 (torch.abs(chunk - maxVal) < 1e-10).double()*gradOutput[:, :, ii:ii+1, jj:jj+1]

		return gradInput



	def clearGrad(self):
		pass

	def dispParam(self):
		pass



if __name__ == '__main__':
	inp = torch.autograd.Variable(torch.rand(32, 3, 28, 28).cuda(), requires_grad=True)
	mP1 = MaxPool(2).cuda()
	mP2 = torch.nn.MaxPool2d(2).cuda()

	out1 = mP1.forward(inp)
	out2 = mP2.forward(inp)

	loss = 2*out2.sum()
	loss.backward()

	gradOut = 2*torch.ones((32,3,14,14), device=inp.device)
	gradIn = mP1.backward(inp, gradOut)

	print(torch.max(torch.abs(out1 - out2)))
	print(torch.max(torch.abs(gradIn - inp.grad)))

	print(inp.grad.shape)
	print(gradIn.shape)
