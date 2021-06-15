from FetchData import FetchData
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten
import numpy as np

dataFetcher = FetchData()
X = np.array(dataFetcher.get_traningset())
y = np.array(dataFetcher.get_traninglabels())

model = Sequential()
model.add(Flatten(input_shape=(28, 28)))
model.add(Dense(128, activation='relu'))
model.add(Dense(10))


model.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer='adam', metrics=['accuracy'])

model.fit(X, y, epochs=10, batch_size=10)

testX = np.array(dataFetcher.get_testset())
testy = np.array(dataFetcher.get_testlabels())

_, accuracy = model.evaluate(testX, testy)
print('Accuracy: %.2f' % (accuracy*100))
