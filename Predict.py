from FetchData import FetchData
import torch
from torch import tensor

class Net:
    def __init__(self, input):
        self.input = input
        self.layer1 = tensor([28,28], dtype= torch.float64)
        print(self.layer1)

dataFetcher = FetchData()
new = Net(dataFetcher.get_traningset())