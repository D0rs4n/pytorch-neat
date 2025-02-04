import torch


def sigmoid(x):
    return torch.sigmoid(x)


def tanh(x):
    return torch.tanh(x)


def relu(x):
    return torch.relu(x)


class Activations:
    def __init__(self):
        self.functions = dict(sigmoid=sigmoid, tanh=tanh, relu=relu)

    def get(self, func_name):
        return self.functions.get(func_name, None)
