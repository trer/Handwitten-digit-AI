from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.utils.np_utils import to_categorical
from tensorflow import keras

from . import DefaultModel


class FlattSequentialModel(DefaultModel.DefaultModel):
    """ Simple Flatt sequential net. """

    def __init__(self):
        super().__init__(loss=keras.losses.CategoricalCrossentropy(
        from_logits=True), optimizer='adam', metrics=['accuracy'])
        self.name = 'FlattSequentialModel'
        self.model = Sequential()
        self.model.add(Flatten(input_shape=(28, 28)))
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dense(10))

    def label_resize(self, data):
        return to_categorical(data)