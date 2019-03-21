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
network['num_layers'] = 3
network['cell_type'] = 'RNN'

config = {'dataset': dataset, 'network': network}

# python3 main.py --model_name first_run --config config.default_config --data_path ../data/ --lr 0.00005 --print_every 10000 --optimizer_type Adam --use_gpu --batch_size 32