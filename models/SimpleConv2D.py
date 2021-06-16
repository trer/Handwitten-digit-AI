from keras.layers import Dense, Conv2D, Flatten
from keras.models import Sequential
from DefaultModel import DefaultModel


class SimpleConv2D(DefaultModel):
    """ A Conv2D network"""

    def __init__(self):
        self.name = 'SimpleConv2D'
        model = Sequential()
        # add model layers
        model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28, 28, 1)))
        model.add(Conv2D(32, kernel_size=3, activation='relu'))
        model.add(Flatten())
        model.add(Dense(10))
        self.model = model

    def resize(self, data):
        return data.reshape(data.shape[0], data.shape[1], data.shape[2], 1)
