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
		output = torch.zeros(N, C, H_out, W_out, device=input.device)

		for ii in range(H_out):
			for jj in range(W_out):
				inp_ii = self.kernel_size*ii
				inp_jj = self.kernel_size*jj
				chunk = input[:, :, inp_ii:inp_ii+self.kernel_size, inp_jj:inp_jj+self.kernel_size]
				output[:, :, ii, jj] = torch.max(torch.max(chunk, 2)[0], 2)[0]

		return output


	def backward(self, input, gradOutput):
		'''
		(N, C, H, W)
		'''
		gradInput = torch.zeros_like(input, device=input.device)

		N, C, H, W = input.shape
		H_out = H//self.kernel_size
		W_out = W//self.kernel_size

		for ii in range(H_out):
			for jj in range(W_out):
				inp_ii = self.kernel_size*ii
				inp_jj = self.kernel_size*jj
				chunk = input[:, :, inp_ii:inp_ii+self.kernel_size, inp_jj:inp_jj+self.kernel_size]
				# chunk is of size N * C * K * K
				maxVal = torch.max(torch.max(chunk, 2)[0], 2)[0][:, :, None, None]
				gradInput[:, :, inp_ii:inp_ii+self.kernel_size, inp_jj:inp_jj+self.kernel_size] =\
						 (torch.abs(chunk - maxVal) < 1e-10).double()*gradOutput[:, :, ii:ii+1, jj:jj+1]

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

	gradOut = 2*torch.ones(inp.shape, device=inp.device)
	gradIn = mP1.backward(inp, gradOut)

	print(torch.max(torch.abs(out1 - out2)))
	print(torch.max(torch.abs(gradIn - inp.grad)))

	print(inp.grad.shape)
	print(gradIn.shape)
