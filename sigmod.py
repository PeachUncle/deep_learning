import numpy as np

def basic_sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = basic_sigmoid(x)
    return s * (1 - s)