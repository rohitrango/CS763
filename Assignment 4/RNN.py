import torch
import math

torch.set_default_dtype(torch.double)

class RNN:

	def __init__(self, num_in, num_hidden, num_out, activation=torch.tanh):
		
		"""
			num_in = size of the one-hot encoded input "word". One element of such a batch will have many such "words".
			Xavier initialization of the weights, the gradients are calculated anyway, so initialization doesn't matter
		"""
		
		self.Wxh = torch.randn(num_hidden, num_in)
		self.Wxh = self.Wxh * math.pow(2 / (num_in + num_hidden), 0.5)
		self.Bxh = torch.zeros(num_hidden, 1)

		self.Whh = torch.randn(num_hidden, num_hidden)
		self.Whh = self.Whh * math.pow(2 / (num_hidden + num_hidden), 0.5)
		self.Bhh = torch.zeros(num_hidden, 1)

		self.Why = torch.randn(num_out, num_hidden)
		self.Why = self.Why * math.pow(2 / (num_hidden + num_out), 0.5)
		self.Bhy = torch.zeros(num_out, 1)

		self.activation = activation

		self.gradWxh = torch.zeros_like(self.Wxh)
		self.gradBxh = torch.zeros_like(self.Bxh)

		self.gradWhh = torch.zeros_like(self.Whh)
		self.gradBhh = torch.zeros_like(self.Bhh)

		self.gradWhy = torch.zeros_like(self.Why)
		self.gradBhy = torch.zeros_like(self.Bhy)

	def cuda(self):
		
		"""
			For transferring to GPU device
		"""
		
		self.Wxh = self.Wxh.cuda()
		self.Bxh = self.Bxh.cuda()

		self.Whh = self.Whh.cuda()
		self.Bhh = self.Bhh.cuda()
		
		self.Why = self.Why.cuda()
		self.Bhy = self.Bhy.cuda()
		
		self.gradWxh = self.gradWxh.cuda()
		self.gradBxh = self.gradBxh.cuda()

		self.gradWhh = self.gradWhh.cuda()
		self.gradBhh = self.gradBhh.cuda()

		self.gradWhy = self.gradWhy.cuda()
		self.gradBhy = self.gradBhy.cuda()

		return self

	def forward(self, input):

		"""
			Assuming input is (batch_size, seq_len, num_input) and output is required to be (batch_size, num_out)
			Assuming within a batch we have a fixed length of sequences, i.e. seq_len
			Hidden state initialised to 0s afresh before each training batch to reduce inter-data dependency
		"""

		batch_size   = input.shape[0]
		seq_length	 = input.shape[1]
		hid_length 	 = self.Bhh.shape[0]
		out_length   = self.Why.shape[0]

		self.hidden_state = torch.zeros(batch_size, seq_length, hid_length).to(self.Wxh.device)
		self.output = torch.zeros(batch_size, seq_length, out_length).to(self.Wxh.device)

		for seq in range(seq_len):
			bat_seq_inp  = input[:, seq, :]
			prev_hidden  = self.hidden_state[:, max(0, seq-1)]
			self.hid_inp = torch.matmul(bat_seq_inp, torch.t(self.Wxh)) + self.Bxh.t()
			self.hid_hid = torch.matmul(prev_hidden, torch.t(self.Whh)) + self.Bhh.t()
			self.hid_tot = self.hid_inp + self.hid_hid

			# Can use ReLU instead ?
			self.hidden_state[:, seq, :] = self.activation(self.hid_tot)
			self.output[:, seq, :] = torch.matmul(self.hidden_state[:, seq], torch.t(self.Why)) + self.Bhy.t()
			# self.hidden_state[seq] = torch.max(0, self.hid_tot)

		output = self.output[:, seq_length-1, :] + 0
		return output


	def backward(self, input, gradOutput):
		
		"""
			gradOutput is (batch_size, seq_len, num_out)
		
			If we only apply a loss to the last layer, then all other gradOuts will be 0
			Since only the last term contributes to the loss
		"""

		seq_length  = self.hidden_state.shape[1]
		inp 		= self.hidden_state[:, seq_length-1, :]	# B *

		self.gradWhy = torch.t(torch.matmul(torch.t(inp), gradOutput[:, seq_length-1]))
		self.gradBhy = torch.t(torch.sum(gradOutput[:, seq_length-1], dim=0).unsqueeze(0))
		gradInput 	 = torch.matmul(gradOutput[:, seq_length-1], self.Why)

		self.gradWxh = torch.zeros_like(self.Wxh)
		self.gradBxh = torch.zeros_like(self.Bxh)

		self.gradWhh = torch.zeros_like(self.Whh)
		self.gradBhh = torch.zeros_like(self.Bhh)

		gradOut = gradInput + 0

		for seq in range(seq_length-1,-1,-1):
			## First differentiating wrt the activation function, assuming tanh here
			inp = self.hidden_state[:, seq, :]
			grad_tanh = 1 - inp**2
			gradOut = grad_tanh * gradOut

			inp = torch.zeros(batch_size, hidden_state)
			if seq > 0:
				inp = self.hidden_state[:, seq, :]
			
			self.gradWhh += torch.t(torch.matmul(torch.t(inp), gradOut))
			self.gradBhh += torch.t(torch.sum(gradOut, dim=0).unsqueeze(0))

			inp = input[:, seq, :]

			self.gradWxh += torch.t(torch.matmul(torch.t(inp), gradOut))
			self.gradBxh += torch.t(torch.sum(gradOut, dim=0).unsqueeze(0))

			gradInput = torch.matmul(gradOutput, self.Whh)
			gradOut = gradInput + 0

		return gradInput