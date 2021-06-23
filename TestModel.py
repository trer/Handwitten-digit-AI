import random
import time

from scipy.ndimage import rotate
from tensorflow import keras
from FetchData import FetchData
from models import DefaultModel
from models import FlattSequentialNet, SimpleConv2D
import numpy as np
import matplotlib.pyplot as plt


def test_model(model, training_set, training_labels, test_set, test_labels):
    """ Tests the model on a labeled dataset split into training- and test-set.
    Returns accuracy of the model and different time measurements. """
    training_set = model.resize(training_set)
    training_labels = model.label_resize(training_labels)
    test_set = model.resize(test_set)
    test_labels = model.label_resize(test_labels)

    model1 = model.model
    t0 = time.time()
    # keras.losses.SparseCategoricalCrossentropy(
    #         from_logits=True)
    model1.compile(loss=model.loss, optimizer=model.optimizer, metrics=model.metrics)
    t1 = time.time()

    model1.fit(training_set, training_labels, epochs=1)
    t2 = time.time()

    _, accuracy = model1.evaluate(test_set, test_labels)
    t3 = time.time()
    model1.save(f"{model.name}.h5")

    compile_time = t1 - t0
    training_time = t2 - t1
    prediction_time = t3 - t2
    print(
        f"accuracy: {accuracy}, compiletime: {compile_time}, trainingTime:"
        f" {training_time}, predictionTime: {prediction_time}")
    return [accuracy, compile_time, training_time, prediction_time]


def plot(
    data_to_plot,
    titles=(
        'accuracy',
        'compile_time',
        'training_time',
        'prediction_time'),
        labels=(
            'FlattSequential',
        'SimpleConv2d')):
    """ Takes in results from benchmarking, benchmarking titles and models tested.
    Plots it to the screen using PyPlot. """
    # create plot
    fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(nrows=2, ncols=2)
    bar_width = 0.35
    opacity = 0.8

    # Add data to separate lists so it can be compared against each other.
    accuracy = []
    compile_time = []
    training_time = []
    prediction_time = []

    for result in data_to_plot:
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


if __name__ == '__main__':
    a = DefaultModel

    # Add all models to a list,
    models = [SimpleConv2D.SimpleConv2D(),
              FlattSequentialNet.FlattSequentialModel()]
    names = [model.name for model in models]

    # Acquire test data.
    dataFetcher = FetchData()
    X = np.array(dataFetcher.get_training_set())
    #X = rotate(X, random.randint(0, 359), axes=(2, 1), reshape=False)
    #fig, axs = plt.subplots(2, 5)
    #for i in range(10):
    #    axs[i // 5, i % 5].imshow(X[i])
    #plt.show()
    y = np.array(dataFetcher.get_training_labels())
    testX = np.array(dataFetcher.get_test_set())
    testy = np.array(dataFetcher.get_test_labels())

    # For each model check accuracy and different time measurements
    results = []
    for m in models:
        results.append(test_model(m, X, y, testX, testy))

    # Plot in a nice graph
    plot(results, labels=names)
