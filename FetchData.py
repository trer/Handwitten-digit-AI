import torchvision.datasets as datasets


class FetchData:
    """ class for getting the dataset loaded from files. """

    def __init__(self):
        pass

    @staticmethod
    def get_training_set():
        return datasets.MNIST(
            root='./data',
            train=True,
            download=True,
            transform=None).train_data

    @staticmethod
    def get_training_labels():
        return datasets.MNIST(
            root='./data',
            train=True,
            download=True,
            transform=None).train_labels

    @staticmethod
    def get_test_set():
        return datasets.MNIST(
            root='./data',
            train=False,
            download=True,
            transform=None).test_data

    @staticmethod
    def get_test_labels():
        return datasets.MNIST(
            root='./data',
            train=False,
            download=True,
            transform=None).test_labels
