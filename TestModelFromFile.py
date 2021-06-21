import time

from keras.models import load_model
from FetchData import FetchData
from models import DefaultModel
from models import FlattSequentialNet, SimpleConv2D
import numpy as np
import matplotlib.pyplot as plt


def test_model(model, test_set, test_labels):
    """ Tests the model on a labeled dataset split into training- and test-set.
    Returns accuracy of the model and different time measurements. """
    test_set = model.resize(test_set)

    # load model
    model1 = load_model(f'{model.name}.h5')
    # summarize model.
    model1.summary()
    t0 = time.time()

    _, accuracy = model1.evaluate(test_set, test_labels)
    t1 = time.time()
    model1.save(f"{model.name}.h5")
    prediction_time = t1 - t0
    print(
        f"accuracy: {accuracy}, predictionTime: {prediction_time}")
    return [accuracy, prediction_time]


def plot(
    data_to_plot,
    titles=(
        'accuracy',
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
    prediction_time = []

    for result in data_to_plot:
        accuracy.append(result[0])
        prediction_time.append(result[1])

    ax0.bar(labels, accuracy, bar_width,
            alpha=opacity)
    ax0.set_title(titles[0])

    ax3.bar(labels, prediction_time, bar_width,
            alpha=opacity)
    ax3.set_title(titles[1])

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
    y = np.array(dataFetcher.get_training_labels())
    testX = np.array(dataFetcher.get_test_set())
    testy = np.array(dataFetcher.get_test_labels())

    # For each model check accuracy and different time measurements
    results = []
    for m in models:
        results.append(test_model(m, testX, testy))

    # Plot in a nice graph
    plot(results, labels=names)
