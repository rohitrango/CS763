"""
Specifies default parameters used for training in a dictionary
"""

config = {}

dataset = {}
dataset['seq_max_len'] = 3000
dataset['num_classes'] = 2
dataset['pad_beginning'] = True

network = {}
network['hidden_size'] = 256
network['num_layers'] = 1
network['cell_type'] = 'RNN'

config = {'dataset': dataset, 'network': network}