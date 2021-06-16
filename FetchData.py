from random import random
from mnist import *
import torchvision.datasets as datasets

class FetchData:
    """ class for getting the dataset loaded from files. """
    def __init__(self):
        pass

    def get_traningset(self):
        return datasets.MNIST(root='./data', train=True, download=True, transform=None).train_data

    def get_traninglabels(self):
        return datasets.MNIST(root='./data', train=True, download=True, transform=None).train_labels

    def get_testset(self):
        return datasets.MNIST(root='./data', train=False, download=True, transform=None).test_data

    def get_testlabels(self):
        return datasets.MNIST(root='./data', train=False, download=True, transform=None).test_labels

