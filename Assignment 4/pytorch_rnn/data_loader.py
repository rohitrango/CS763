"""
Interface to load data from file and handle variable length data while batching
"""

import sys
import os

import numpy as np
import torch
import torch.utils.data

def pad_tensor(v, pad, pad_tensor, truncate_end=True):
    """
    inputs:
        v: 2D torch tensor
        pad: the final length required along dim=0
        pad_tensor: the tensor used to pad
        truncate_end: used when length of sequence is greater than 'pad'.
            if true, then truncates the last part to reduce length
            else, truncates the first part to reduce length
    return:
        returns the tensor of dim=0 'pad' obtained by padding/truncating v 
    """

    if (v.size(0) > pad):
        if (truncate_end):
            return v[: pad]
        else:
            return v[-pad :]
    elif (v.size(0) < pad):
        return torch.cat([v, pad_tensor.repeat(pad - v.size(0), 1)], dim=0)
    else:
        return v

class PadCollate:
    """
    Class used for to create object to be passed as collate_fn to make sequences of equal length in dataloader in a batch
    """
    def __init__(self, max_length, pad_tensor=None, truncate_end=True):
        """
        input:
            max_length: the maximum length of the padded sequence
            pad_tensor: the tensor to use for padding
            truncate_end: used when a sequence of length > 'max_length' is encountered
                if true, then truncates the last part to reduce length
                else, truncates the first part to reduce length
        """
        self.max_length = max_length
        self.pad_tensor = pad_tensor
        self.truncate_end = truncate_end


    def pad_collate(self, batch):
        """
        input:
            batch: list of (x, y) of the batch
        return:
            batch_xs, batch_ys after padding x to make them of equal length
            batch_xs: tensor of size (T, batch_size, input_size)
            batch_ys: torch LongTensor of size (batch_size, )
        """
        if (self.pad_tensor is None):
            self.pad_tensor = torch.zeros(batch[0][0].size(1))

        pad = max(map(lambda x: x[0].size(0), batch))
        if (pad > self.max_length):
            pad = self.max_length

        batch = list(map(lambda x: (pad_tensor(x[0], pad=pad, pad_tensor=self.pad_tensor, truncate_end=self.truncate_end), x[1]), batch))
        batch_xs = torch.stack(list(map(lambda x: x[0], batch)), dim=1)
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

class ListDataset(torch.utils.data.Dataset):
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
        x = torch.stack([one_hot_tensor(self.word_to_index[x], len(self.word_to_index)) for x in self.X[index]], dim=0)
        return x, self.y[index]

    def __len__(self):
        return len(self.X)


def load_train_data(data_path):
    """
    loads train data from the path given
    input:
        path: path to the folder containing two files, train_data.txt and train_labels.txt
    output:
        (X, y)
        where X is list of sentences, where a sentence is a list of words
        y is list of labels
    """
    X, y = [], []
    with open(os.path.join(data_path, 'train_data.txt'), 'r') as f:
        while (True):
            line = f.readline()
            if (line == ''):
                break
            X.append([int(v) for v in line.split()])

    with open(os.path.join(data_path, 'train_labels.txt'), 'r') as f:
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
    with open(os.path.join(data_path, 'train_data.txt'), 'r') as f:
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



if __name__ == '__main__':
    
    # X, y = load_train_data(sys.argv[1])
    # X_test = load_test_data(sys.argv[2])
    # word_to_index = get_word_to_index_dict(X)
    # # word_to_index_test = get_word_to_index_dict(X_test)
    # # X = one_hot(X)

    # data_loader = ListDataset(X, y, word_to_index)


    X = [[1, 2], [1, 2, 3], [1, 2, 3, 4], [1, 2]]
    y = [1, 2, 3, 4]
    word_to_index = get_word_to_index_dict(X)

    data = ListDataset(X, y, word_to_index)

    data_loader = torch.utils.data.DataLoader(data, batch_size=2, collate_fn=PadCollate(2000))

    # for i in a:
    #     print(i[0])
    #     print(i[1])
    #     print()
