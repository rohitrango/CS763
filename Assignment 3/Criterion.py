import torch

torch.set_default_dtype(torch.double)

# Can't use nn package, so how do we create a torch class ? For now, a normal python class.
class Criterion():
	'''
		Implements the Cross Entropy Loss only as of now
	'''
	def __init__(self):
		'''
			Empty function for now
		'''
		pass

	def forward(self, input, target):
		'''
			Assuming input is (batch_size,num_classes) and target is 1D - (batch_size)
			Computes average cross entropy loss over the batch
			Assuming classes are 0 indexed
		'''
		batch_size = input.size(0)
		target = target.long()
		neg_scores = -input[range(batch_size),target]
		log_sum_scores = torch.log(torch.sum(torch.exp(input),1))
		losses = neg_scores+log_sum_scores
		loss = torch.sum(losses)/batch_size
		return loss

	def backward(self, input, target):
		'''
			Returns gradient of the loss wrt input
		'''
		batch_size = input.size(0)
		target = target.long()
		target_batch = torch.zeros_like(input)
		target_batch[range(batch_size),target] = 1
		ex_inp = torch.exp(input)
		sum_scores = torch.sum(ex_inp,1)
		sum_scores = torch.t(sum_scores.unsqueeze(0))
		probs = torch.exp(input)/sum_scores
		gradient = probs - target_batch
		return gradient