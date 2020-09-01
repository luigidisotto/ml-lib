from scipy.special import expit


class Sigmoid(object):
    @staticmethod
    def apply(x):
        return expit(x)

    @staticmethod
    def derivative(x):
        return expit(x)*(1.0 - expit(x))


class Linear(object):
    @staticmethod
    def apply(x):
        return x

    @staticmethod
    def derivative(x):
        return 1.0
