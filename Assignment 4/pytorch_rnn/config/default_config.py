"""
Specifies default parameters used for training in a dictionary
"""

config = {}

dataset = {}
dataset['seq_max_len'] = 3000
dataset['num_classes'] = 2

network = {}
network['hidden_size'] = 64
network['num_layers'] = 1

config = {'dataset': dataset, 'network': network}