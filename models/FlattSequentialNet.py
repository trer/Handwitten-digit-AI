from keras.models import Sequential
from keras.layers import Dense, Flatten
from . import DefaultModel


class FlattSequentialModel(DefaultModel.DefaultModel):
    """ Simple Flatt sequential net. """

    def __init__(self):
        self.name = 'FlattSequentialModel'
        self.model = Sequential()
        self.model.add(Flatten(input_shape=(28, 28)))
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dense(10))
