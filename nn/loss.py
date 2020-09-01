import numpy as np


class MeanSquaredError(object):
    @staticmethod
    def apply(t, o):
        return 0.5*np.linalg.norm((o - t), 2)**2

    @staticmethod
    def derivative(t, o):
        return (o - t)


class MeanEuclideanError(object):
    @staticmethod
    def apply(t, o):
        """Returns the Mean Euclidean Error associated with an output ``o`` and desired output
        ``t``.
        MEMENTO: multiply by 0.5 if you would use it as loss function
        """
        return np.linalg.norm((o - t), 2)

    @staticmethod
    def derivative(t, o):
        """Return the error gradient from the output layer."""
        return (o - t)/np.linalg.norm((o - t), 2)
