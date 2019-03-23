import torch

torch.set_default_dtype(torch.double)

class CrossEntropyLoss():
	'''
		Implements the Cross Entropy Loss
	'''
	def __init__(self):
		'''
			Empty function for now
		'''
		pass

	def cuda(self):
		return self

	def forward(self, input, target):
		'''
			Assuming input is (batch_size,num_classes) and target is 1D - (batch_size)
			Computes average cross entropy loss over the batch
			Assuming classes are 0 indexed
		'''
		batch_size = input.size(0)
		target = target.long()
		neg_scores = -(input[range(batch_size),target] - torch.max(input, dim=1)[0])
		log_sum_scores = torch.log(torch.sum(torch.exp(input - torch.max(input, dim=1, keepdim=True)[0]), dim=1))
		losses = neg_scores + log_sum_scores
		loss = torch.mean(losses)
		return loss

	def backward(self, input, target):
		'''
			Returns gradient of the loss wrt input
		'''
		batch_size = input.size(0)
		target = target.long()
		target_batch = torch.zeros_like(input, device=input.device)
		target_batch[range(batch_size),target] = 1
		ex_inp = torch.exp(input - torch.max(input, dim=1, keepdim=True)[0])
		sum_scores = torch.sum(ex_inp, dim=1, keepdim=True)
		probs = torch.exp(input - torch.max(input, dim=1, keepdim=True)[0]) / sum_scores
		gradient = (probs - target_batch) / batch_size
		return gradient