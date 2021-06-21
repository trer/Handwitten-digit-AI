

class DefaultModel:
    """ Model parent"""

    def resize(self, data):
        """ returns the data reshaped to fit to this model.
        As models have different variations on how they want the data to be represented, this can be fixed here.
        """
        return data
