from keras.models import Sequential
from keras.layers import Dense, Flatten


class FlattSequentialModel:

    def __init__(self):
        self.name = 'FlattSequentialModel'
        self.model = Sequential()
        self.model.add(Flatten(input_shape=(28, 28)))
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dense(10))

    def resize(self, data):
        return data
