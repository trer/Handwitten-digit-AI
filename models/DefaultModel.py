

class DefaultModel:
    """ Model parent"""

    def __init__(self, loss=None, optimizer=None, metrics=None):
        self.loss = loss
        self.optimizer = optimizer
        self.metrics = metrics

    def resize(self, data):
        """ returns the inputdata reshaped to fit to this model.
        As models have different variations on how they want the data to be represented, this can be fixed here.
        """
        return data

    def label_resize(self, data):
        """ returns the ground_truth reshaped to fit this model. """
        return data