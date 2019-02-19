import torch
import torch.nn as nn
import numpy as np

class Network(nn.Module):
	def __init__(self, input_shape, output_shape, dropout=(0.0, 0.0)):
		super(Network, self).__init__()
		input_size = np.prod(input_shape)
		self.layers = []
		self.layers.append(nn.Dropout(p=dropout[0]))
		self.layers.append(nn.Linear(input_size, 200))
		self.layers.append(nn.ReLU())
		self.layers.append(nn.Dropout(p=dropout[1]))
		self.layers.append(nn.Linear(200, 100))
		self.layers.append(nn.ReLU())
		self.layers.append(nn.Dropout(p=dropout[1]))
		self.layers.append(nn.Linear(100, output_shape[0]))
		self.layers = nn.Sequential(*tuple(self.layers))

	def forward(self, x):
		y = x
		y = torch.reshape(y, shape=(y.shape[0], -1))
		y = self.layers(y)
		return y

class BNNetwork(nn.Module):
	def __init__(self, input_shape, output_shape):
		super(BNNetwork, self).__init__()
		input_size = np.prod(input_shape)
		self.layers = []
		self.layers.append(nn.Linear(input_size, 200))
		self.layers.append(nn.ReLU())
		self.layers.append(nn.BatchNorm1d(200))
		self.layers.append(nn.Linear(200, 200))
		self.layers.append(nn.ReLU())
		self.layers.append(nn.BatchNorm1d(200))
		self.layers.append(nn.Linear(200, 200))
		self.layers.append(nn.ReLU())
		self.layers.append(nn.BatchNorm1d(200))
		self.layers.append(nn.Linear(200, 200))
		self.layers.append(nn.ReLU())
		self.layers.append(nn.BatchNorm1d(200))
		self.layers.append(nn.Linear(200, 200))
		self.layers.append(nn.ReLU())
		self.layers.append(nn.BatchNorm1d(200))
		self.layers.append(nn.Linear(200, output_shape[0]))
		self.layers = nn.Sequential(*tuple(self.layers))

	def forward(self, x):
		y = x
		y = torch.reshape(y, shape=(y.shape[0], -1))
		y = self.layers(y)
		return y

class BNConvNetworkSmall(nn.Module):
	def __init__(self, input_shape, output_shape):
		super(BNConvNetworkSmall, self).__init__()
		input_size = np.prod(input_shape)
		self.conv_layers = []
		self.linear_layers = []
		self.conv_layers.append(nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1))
		self.conv_layers.append(nn.ReLU())
		self.conv_layers.append(nn.BatchNorm2d(16))
		self.conv_layers.append(nn.MaxPool2d(kernel_size=2))

		self.conv_layers.append(nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3))
		self.conv_layers.append(nn.ReLU())
		self.conv_layers.append(nn.BatchNorm2d(16))
		self.conv_layers.append(nn.MaxPool2d(kernel_size=2))

		self.conv_layers.append(nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3))
		self.conv_layers.append(nn.ReLU())
		self.conv_layers.append(nn.BatchNorm2d(16))
		self.conv_layers.append(nn.MaxPool2d(kernel_size=6))

		# self.conv_layers.append(nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1))
		# self.conv_layers.append(nn.ReLU())
		# self.conv_layers.append(nn.BatchNorm2d(32))
		# self.conv_layers.append(nn.MaxPool2d(kernel_size=6))

		# self.conv_layers.append(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1))
		# self.conv_layers.append(nn.ReLU())
		# self.conv_layers.append(nn.BatchNorm2d(64))
		# self.conv_layers.append(nn.MaxPool2d(kernel_size=6))

		# self.linear_layers.append(nn.Linear(64, 200))
		# self.linear_layers.append(nn.ReLU())
		# self.linear_layers.append(nn.BatchNorm1d(200))
		# self.linear_layers.append(nn.Linear(200, 100))
		# self.linear_layers.append(nn.ReLU())
		# self.linear_layers.append(nn.BatchNorm1d(100))
		self.linear_layers.append(nn.Linear(16 * 4 * 4, output_shape[0]))
		
		self.conv_layers = nn.Sequential(*tuple(self.conv_layers))
		self.linear_layers = nn.Sequential(*tuple(self.linear_layers))

	def forward(self, x):
		y = x
		y = self.conv_layers(y)
		y = torch.reshape(y, shape=(y.shape[0], -1))
		y = self.linear_layers(y)
		return y


class ConvNetwork(nn.Module):
	def __init__(self, input_shape, output_shape, dropout=(0.0, 0.0)):
		super(ConvNetwork, self).__init__()
		input_size = np.prod(input_shape)
		self.conv_layers = []
		self.linear_layers = []
		self.conv_layers.append(nn.Dropout(p=dropout[0]))
		self.conv_layers.append(nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1))
		self.conv_layers.append(nn.ReLU())
		self.conv_layers.append(nn.Dropout(p=dropout[1]))
		self.conv_layers.append(nn.MaxPool2d(kernel_size=2))

		self.conv_layers.append(nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3))
		self.conv_layers.append(nn.ReLU())
		self.conv_layers.append(nn.Dropout(p=dropout[1]))
		self.conv_layers.append(nn.MaxPool2d(kernel_size=2))

		self.conv_layers.append(nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3))
		self.conv_layers.append(nn.ReLU())
		self.conv_layers.append(nn.Dropout(p=dropout[1]))
		self.conv_layers.append(nn.MaxPool2d(kernel_size=2))

		self.conv_layers.append(nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1))
		self.conv_layers.append(nn.ReLU())
		self.conv_layers.append(nn.Dropout(p=dropout[1]))
		self.conv_layers.append(nn.MaxPool2d(kernel_size=2))

		self.conv_layers.append(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1))
		self.conv_layers.append(nn.ReLU())
		self.conv_layers.append(nn.Dropout(p=dropout[1]))
		self.conv_layers.append(nn.MaxPool2d(kernel_size=6))


		self.linear_layers.append(nn.Linear(64, 200))
		self.linear_layers.append(nn.ReLU())
		self.linear_layers.append(nn.Dropout(p=dropout[1]))
		self.linear_layers.append(nn.Linear(200, 100))
		self.linear_layers.append(nn.ReLU())
		self.linear_layers.append(nn.Dropout(p=dropout[1]))
		self.linear_layers.append(nn.Linear(100, output_shape[0]))
		
		self.conv_layers = nn.Sequential(*tuple(self.conv_layers))
		self.linear_layers = nn.Sequential(*tuple(self.linear_layers))

	def forward(self, x):
		y = x
		y = self.conv_layers(y)
		y = torch.reshape(y, shape=(y.shape[0], -1))
		y = self.linear_layers(y)
		return y

class BNConvNetwork(nn.Module):
	def __init__(self, input_shape, output_shape):
		super(BNConvNetwork, self).__init__()
		input_size = np.prod(input_shape)
		self.conv_layers = []
		self.linear_layers = []
		self.conv_layers.append(nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1))
		self.conv_layers.append(nn.ReLU())
		self.conv_layers.append(nn.BatchNorm2d(16))
		self.conv_layers.append(nn.MaxPool2d(kernel_size=2))

		self.conv_layers.append(nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3))
		self.conv_layers.append(nn.ReLU())
		self.conv_layers.append(nn.BatchNorm2d(16))
		self.conv_layers.append(nn.MaxPool2d(kernel_size=2))

		self.conv_layers.append(nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3))
		self.conv_layers.append(nn.ReLU())
		self.conv_layers.append(nn.BatchNorm2d(16))
		self.conv_layers.append(nn.MaxPool2d(kernel_size=2))

		self.conv_layers.append(nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1))
		self.conv_layers.append(nn.ReLU())
		self.conv_layers.append(nn.BatchNorm2d(32))
		self.conv_layers.append(nn.MaxPool2d(kernel_size=2))

		self.conv_layers.append(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1))
		self.conv_layers.append(nn.ReLU())
		self.conv_layers.append(nn.BatchNorm2d(64))
		self.conv_layers.append(nn.MaxPool2d(kernel_size=6))


		self.linear_layers.append(nn.Linear(64, 200))
		self.linear_layers.append(nn.ReLU())
		self.linear_layers.append(nn.BatchNorm1d(200))
		self.linear_layers.append(nn.Linear(200, 100))
		self.linear_layers.append(nn.ReLU())
		self.linear_layers.append(nn.BatchNorm1d(100))
		self.linear_layers.append(nn.Linear(100, output_shape[0]))
		
		self.conv_layers = nn.Sequential(*tuple(self.conv_layers))
		self.linear_layers = nn.Sequential(*tuple(self.linear_layers))

	def forward(self, x):
		y = x
		y = self.conv_layers(y)
		y = torch.reshape(y, shape=(y.shape[0], -1))
		y = self.linear_layers(y)
		return y