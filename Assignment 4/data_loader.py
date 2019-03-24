"""
Interface to load data from file and handle variable length data while batching
"""

import sys
import os
import random
import math

import numpy as np
import torch

def pad_tensor(v, pad, pad_tensor, truncate_end=True, pad_beginning=False):
    """
    inputs:
        v: 2D torch tensor
        pad: the final length required along dim=0
        pad_tensor: the tensor used to pad
        truncate_end: used when length of sequence is greater than 'pad'.
            if true, then truncates the last part to reduce length
            else, truncates the first part to reduce length
        pad_beginning: if True, then pad PAD in the beginning, else pad PAD in the end
    return:
        returns the tensor of dim=0 'pad' obtained by padding/truncating v 
    """

    if (v.size(0) > pad):
        if (truncate_end):
            return v[: pad]
        else:
            return v[-pad :]
    elif (v.size(0) < pad):
        if (pad_beginning):
            return torch.cat([pad_tensor.repeat(pad - v.size(0), 1), v], dim=0)
        else:
            return torch.cat([v, pad_tensor.repeat(pad - v.size(0), 1)], dim=0)

    else:
        return v

class PadCollate:
    """
    Class used for to create object to be passed as collate_fn to make sequences of equal length in dataloader in a batch
    """
    def __init__(self, max_length, pad_tensor=None, truncate_end=True, pad_beginning=False):
        """
        input:
            max_length: the maximum length of the padded sequence
            pad_tensor: the tensor to use for padding
            truncate_end: used when a sequence of length > 'max_length' is encountered
                if true, then truncates the last part to reduce length
                else, truncates the first part to reduce length
            pad_beginning: if True, then pad PAD in the beginning, else pad PAD in the end
        """
        self.max_length = max_length
        self.pad_tensor = pad_tensor
        self.truncate_end = truncate_end
        self.pad_beginning = pad_beginning


    def pad_collate(self, batch):
        """
        input:
            batch: list of (x, y) of the batch
        return:
            batch_xs, batch_ys after padding x to make them of equal length
            batch_xs: tensor of size (batch_size, T, input_size)
            batch_ys: torch LongTensor of size (batch_size, )
        """
        if (self.pad_tensor is None):
            self.pad_tensor = torch.zeros(batch[0][0].size(1))

        pad = max(map(lambda x: x[0].size(0), batch))
        if (pad > self.max_length):
            pad = self.max_length

        batch = list(map(lambda x: (pad_tensor(x[0], pad=pad, pad_tensor=self.pad_tensor, truncate_end=self.truncate_end, pad_beginning=self.pad_beginning), x[1]), batch))
        batch_xs = torch.stack(list(map(lambda x: x[0], batch)), dim=0)
        batch_ys = torch.LongTensor(list(map(lambda x: x[1], batch)))
        return batch_xs, batch_ys

    def __call__(self, batch):
        return self.pad_collate(batch)

def one_hot_tensor(index, size):
    """
    Creates a one hot torch tensor
    input:
        index: position which should be 1
        size: size of the tensor
    output:
        torch tensor 'vec' of size 'size' and vec[index] = 1
    """
    vec = torch.zeros(size)
    vec[index] = 1
    return vec

class ListDataset:
    """
    Dataset for storing train data.
    For index access, returns (one-hot encoded train input, train target)
    """
    def __init__(self, X, y, word_to_index):
        """
        input:
            X: list of sentences, where a sentence is a list of words
            y: list of labels for each sentence
            word_to_index: dictionary giving unique position in one-hot vector for each word
        """
        self.X = X
        self.y = y
        self.word_to_index = word_to_index

    def __getitem__(self, index):
        """
        Used for [] access
        input:
            index: position of dataset to access
        output:
            (x, y)
            where x is one-hot encoded tensor of size (seq_len, encoding_size), y is target label
        """
        x = torch.stack([one_hot_tensor(self.word_to_index.get(x, self.word_to_index['OOV']), len(self.word_to_index)) for x in self.X[index]], dim=0)
        return x, self.y[index]

    def __len__(self):
        return len(self.X)


def load_train_data(data_path, labels_path):
    """
    loads train data from the path given
    input:
        data_path: path to train_data.txt
        labels_path: path to train_labels.txt
    output:
        (X, y)
        where X is list of sentences, where a sentence is a list of words
        y is list of labels
    """
    X, y = [], []
    with open(data_path, 'r') as f:
        while (True):
            line = f.readline()
            if (line == ''):
                break
            X.append([int(v) for v in line.split()])

    with open(labels_path, 'r') as f:
        while (True):
            line = f.readline()
            if (line == ''):
                break
            y.append(int(line))

    return X, y

def load_test_data(data_path):
    """
    loads test data from the path given
    input:
        path: path to the folder containing test_data.txt
    output:
        X
        where X is list of sentences, where a sentence is a list of words
    """
    X = []
    with open(data_path, 'r') as f:
        while (True):
            line = f.readline()
            if (line == ''):
                break
            X.append([int(v) for v in line.split()])

    return X

def get_word_to_index_dict(X):
    """
    input:
        X: train input dataset, list of sentences, where a sentence is a list of words
    returns:
        dictionary using which words can be converted to a unique index in one-hot vector
    """
    word_to_index = {}
    index = 0
    for i in range(len(X)):
        for j in range(len(X[i])):
            if (X[i][j] not in word_to_index):
                word_to_index[X[i][j]] = index
                index += 1

    word_to_index['PAD'] = index
    index += 1
    word_to_index['OOV'] = index
    index += 1

    return word_to_index

def load_train_val_dataset(data_path, labels_path, fraction_validation):
    # load and shuffle train dataset
    X_train, y_train = load_train_data(data_path, labels_path)
    word_to_index = get_word_to_index_dict(X_train)
    train_data = list(zip(X_train, y_train))
    random.shuffle(train_data)
    X_train[:], y_train[:] = zip(*train_data)

    num_validation = int(len(X_train) * fraction_validation)
    if (num_validation == 0):
        print('Gives zero number of validation examples')
        X_val, y_val = None, None
    else:
        X_val, y_val = X_train[0 : num_validation], y_train[0 : num_validation]

    X_train, y_train = X_train[num_validation : ], y_train[num_validation : ]
    
    # train_data = list(zip(X_train, y_train))
    # train_data = sorted(train_data, key=lambda x: len(x[0]))
    # X_train, y_train = zip(*train_data)

    return (X_train, y_train), (X_val, y_val), word_to_index

class DataLoader:
    def __init__(self, dataset, batch_size, collate_fn=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.collate_fn = collate_fn
        self.pos = 0
        self.num_batches = math.ceil(len(dataset) / batch_size)
    
    def next_batch(self):
        if (self.pos + self.batch_size >= len(self.dataset)):
            res = [self.dataset[i] for i in range(self.pos, len(self.dataset))]
            self.pos = len(self.dataset)
            return self.collate_fn(res)
        else:
            res = [self.dataset[i] for i in range(self.pos, self.pos + self.batch_size)]
            self.pos += self.batch_size
            return self.collate_fn(res)

    def is_done_epoch(self):
        if (self.pos == len(self.dataset)):
            return True
        else:
            return False
    
    def reset_pos(self):
        self.pos = 0


if __name__ == '__main__':
    pass
    # X, y = load_train_data(sys.argv[1])
    # X_test = load_test_data(sys.argv[2])
    # word_to_index = get_word_to_index_dict(X)
    # word_to_index_test = get_word_to_index_dict(X_test)
    # X = one_hot(X)

    # data_loader = ListDataset(X, y, word_to_index)


    # X = [[1, 2], [1, 2, 3], [1, 2, 3, 4], [1, 2]]
    # y = [1, 2, 3, 4]
    # word_to_index = get_word_to_index_dict(X)

    # data = ListDataset(X, y, word_to_index)

    # data_loader = torch.utils.data.DataLoader(data, batch_size=2, collate_fn=PadCollate(2000))

    # for i in a:
    #     print(i[0])
    #     print(i[1])
    #     print()
