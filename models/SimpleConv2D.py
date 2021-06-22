from keras.layers import Dense, Conv2D, Flatten
from keras.models import Sequential
from tensorflow import keras

from . import DefaultModel


class SimpleConv2D(DefaultModel.DefaultModel):
    """ A Conv2D network"""

    def __init__(self):
        super().__init__(loss=keras.losses.SparseCategoricalCrossentropy(
            from_logits=True), optimizer='adam', metrics=['accuracy'])
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
