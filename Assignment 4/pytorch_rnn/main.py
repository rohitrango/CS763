"""
Creates torch RNN model and trains it with the given dataset
"""

import argparse
import importlib

import numpy as np
import torch
import torch.utils.data

import data_loader

class RNNManyToOne(torch.nn.Module):
	def __init__(self):
		super(RNNManyToOne, self).__init__()
		self.rnn = torch.nn.RNN(input_size=len(word_to_index), hidden_size=config['network']['hidden_size'], num_layers=config['network']['num_layers'])
		self.lin = torch.nn.Linear(in_features=config['network']['hidden_size'], out_features=config['dataset']['num_classes'])

	def forward(self, x, h0):
		output, hn = self.rnn(x, h0)
		scores = self.lin(output[-1])

		return scores

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', help='name of model; name used to create folder to save model')
parser.add_argument('--config', help='path to file containing config dictionary; path in python module format')
parser.add_argument('--data_path', help='path to folder containing train_data.txt, train_labels.txt, test_data.txt')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--epochs', type=int, default=50, help='number of epochs')
parser.add_argument('--reg', type=float, default=0.0, help='regularization weight')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum in momentum optimizer')
parser.add_argument('--batch_size', type=int, default=128, help='batch size for training, testing')
parser.add_argument('--print_every', type=int, default=1000, help='frequency to print train loss, accuracy to terminal')
parser.add_argument('--fraction_validation', type=float, default=0.1, help='fraction of data to be used for validation')
parser.add_argument('--use_gpu', action='store_true', help='whether to use gpu for training/testing')

args = parser.parse_args()

np.random.seed(0)
torch.manual_seed(0)

cpu_device = torch.device('cpu')
fast_device = torch.device('cpu')
if (args.use_gpu):
	fast_device = torch.device('cuda:0')


config = importlib.import_module(args.config).config

X_train, y_train = data_loader.load_train_data(args.data_path)
X_test = data_loader.load_test_data(args.data_path)
y_test_dummy = [int(0) for _ in range(len(X_test))]
word_to_index = data_loader.get_word_to_index_dict(X_train)

train_dataset = data_loader.ListDataset(X_train, y_train, word_to_index)
test_dataset = data_loader.ListDataset(X_test, y_test_dummy, word_to_index)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=data_loader.PadCollate(config['dataset']['seq_max_len']), shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=data_loader.PadCollate(config['dataset']['seq_max_len']))

model = RNNManyToOne()

model = model.to(fast_device)

criterion = torch.nn.CrossEntropyLoss()
optim = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.reg)

loss = []
acc = []

for epoch in range(args.epochs):
	print('epoch:', epoch)
	i = 0
	for batch in train_loader:
		batch_xs, batch_ys = batch
		batch_xs, batch_ys = batch_xs.to(fast_device), batch_ys.to(fast_device)
		optim.zero_grad()
		h0 = torch.zeros((config['network']['num_layers'], batch_xs.size(1), config['network']['hidden_size']))
		scores = model(batch_xs, h0)
		
		cur_loss = criterion(scores, batch_ys)
		cur_acc = torch.sum(torch.argmax(scores, dim=1).long() == batch_ys.long()).item() * 1.0 / batch_xs.size(1)
		loss.append(cur_loss)
		acc.append(cur_acc) 
		
		cur_loss.backward()
		optim.step()
		
		if (i % args.print_every == 0):
			print("iter: %d, Train loss : %f, Train acc : %f" % (i ,loss[-1], acc[-1]))

		i += 1