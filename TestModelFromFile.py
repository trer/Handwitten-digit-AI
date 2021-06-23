import time

import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
from keras.utils.np_utils import to_categorical
from scipy.ndimage import rotate

from FetchData import FetchData
from models import FlattSequentialNet, SimpleConv2D


def test_model(model, test_set, test_labels):
    """ Tests the model on a labeled dataset split into training- and test-set.
    Returns accuracy of the model and different time measurements. """
    test_set = model.resize(test_set)
    test_labels = model.label_resize(test_labels)

    # load model
    model1 = load_model(f'{model.name}.h5')
    # summarize model.
    model1.summary()
    t0 = time.time()

    _, accuracy = model1.evaluate(test_set, test_labels)
    t1 = time.time()
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


def show_model_preds(model, test_set, test_labels):
    """ Shows model predictions compared to test_labels """
    test_set_for_model = model.resize(test_set)
    test_labels_for_plot = to_categorical(test_labels)

    # load model
    model1 = load_model(f'{model.name}.h5')
    # summarize model
    model1.summary()

    for i in range(len(test_set[0])):
        result = model1.predict(test_set_for_model[i].reshape(1, test_set_for_model.shape[1],
                                                              test_set_for_model.shape[2], 1))

        # Only plot if prediction and ground truth is not the same
        if np.argmax(result) != np.argmax(test_labels_for_plot[i]):
            fig, axs = plt.subplots(1, 3)
            axs[0].imshow(test_set[i])
            axs[1].yticks = 1
            axs[1].text(np.argmax(result), -1, np.argmax(result))
            axs[1].imshow(result)
            axs[2].yticks = 1
            axs[2].text(np.argmax(test_labels_for_plot[i]), -1, np.argmax(test_labels_for_plot[i]))
            axs[2].imshow(test_labels_for_plot[i].reshape(1, test_labels_for_plot.shape[1]))
            plt.show()


if __name__ == '__main__':
    rotate_img = False
    show_model = True

    # Add all models to a list,
    models = [SimpleConv2D.SimpleConv2D(),
              FlattSequentialNet.FlattSequentialModel()]
    names = [model.name for model in models]

    # Acquire test data.
    dataFetcher = FetchData()
    testX = np.array(dataFetcher.get_test_set())
    testy = np.array(dataFetcher.get_test_labels())
    if rotate_img is True:
        fig, axs = plt.subplots(2, 5)
        for i in range(10):
            axs[i // 5, i % 5].imshow(testX[i])
        plt.show()
        testX = rotate(testX, 90, axes=(2, 1))
        fig, axs = plt.subplots(2, 5)
        for i in range(10):
            axs[i // 5, i % 5].imshow(testX[i])
        plt.show()
    # For each model check accuracy and different time measurements

    if show_model:
        for m in models:
            show_model_preds(m, testX, testy)
    else:
        results = []
        for m in models:
            results.append(test_model(m, testX, testy))
        # Plot in a nice graph
        plot(results, labels=names)
