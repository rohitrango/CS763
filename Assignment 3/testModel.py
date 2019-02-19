import argparse
import sys
from Model import Model
from Linear import Linear
from Criterion import Criterion
from optim import MomentumOptimizer
from ReLU import ReLU
import torch
import numpy as np
import torchfile, pickle, os, sys
import utils
import math
import matplotlib.pyplot as plt 						# CHECK : finally remove this package

torch.set_default_dtype(torch.double)

parser = argparse.ArgumentParser()
parser.add_argument('-modelName', help='name of model; name used to create folder to save model')
parser.add_argument('-data', help='path to train data')
parser.add_argument('--batch_size', type=int, default=128, help='batch size for training, testing')

args = parser.parse_args()

input = torchfile.load(args.data)
input = np.reshape(input, (input.shape[0], -1))
input = input.astype(np.float32)
min_val, max_val = 0.0, 255.0
input = (input - min_val) / (max_val - min_val) - 0.5

model = torch.load(os.path.join(args.modelName, 'model.pt'))['model']
pred = utils.getPredictions(model, input, batch_size)

with open(os.path.join(args.modelName, 'test_pred.txt'), 'w') as f:
	f.write('id,label\n')
	for i in range(pred.shape[0]):
		f.write(str(i))
		f.write(',')
		f.write(str(pred[i]))

# INCOMPLETE
# save testPrediction.bin