from tensorflow import keras
from FetchData import FetchData
from models import FlattSequentialNet, SimpleConv2D
import numpy as np
import time
import matplotlib.pyplot as plt

dataFetcher = FetchData()
X = np.array(dataFetcher.get_traningset())
y = np.array(dataFetcher.get_traninglabels())
testX = np.array(dataFetcher.get_testset())
testy = np.array(dataFetcher.get_testlabels())


def test_model(model, training_set, training_labels, test_set, test_labels):
    training_set = model.resize(training_set)
    test_set = model.resize(test_set)

    model = model.model
    t0 = time.time()
    model.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer='adam',
                  metrics=['accuracy'])
    t1 = time.time()

    model.fit(training_set, training_labels, epochs=1, batch_size=10)
    t2 = time.time()

    _, accuracy = model.evaluate(test_set, test_labels)
    t3 = time.time()

    compile_time = t1 - t0
    training_time = t2 - t1
    prediction_time = t3 - t2
    print(
        f"accuracy: {accuracy}, compiletime: {compile_time}, trainingTime: {training_time}, predictionTime: {prediction_time}")
    return [accuracy, compile_time, training_time, prediction_time]


def plot(results, titles = ('accuracy', 'compile_time', 'training_time', 'prediction_time'),
    labels = ('FlattSequential', 'SimpleConv2d')):
    """ ayaya """
    # create plot
    fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(nrows=2, ncols=2)
    axis = [ax0, ax1, ax2, ax3]
    bar_width = 0.35
    opacity = 0.8

    #Data
    accuracy = []
    compile_time = []
    training_time = []
    prediction_time = []

    for result in results:
        accuracy.append(result[0])
        compile_time.append(result[1])
        training_time.append(result[2])
        prediction_time.append(result[3])


    ax0.bar(labels, accuracy, bar_width,
            alpha=opacity)
    ax0.set_title(titles[0])

    ax1.bar(labels, compile_time, bar_width,
            alpha=opacity)
    ax1.set_title(titles[1])

    ax2.bar(labels, training_time, bar_width,
            alpha=opacity)
    ax2.set_title(titles[2])

    ax3.bar(labels, prediction_time, bar_width,
            alpha=opacity)
    ax3.set_title(titles[3])

    fig.tight_layout()
    plt.show()


models = []
model2 = SimpleConv2D.SimpleConv2D()
models.append(model2)
model1 = FlattSequentialNet.FlattSequentialModel()
models.append(model1)
names = [model.name for model in models]

results = []
for model in models:
    results.append(test_model(model, X, y, testX, testy))

plot(results, labels=names)

"""
model = SimpleConv2D.SimpleConv2D()
print(X.shape)
X = model.resize(X)
print(X.shape)
testX = model.resize(testX)

test_model(model, X, y, testX, testy)
"""
