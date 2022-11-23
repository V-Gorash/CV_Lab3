import numpy as np
from scipy.special import softmax


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def dsigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))


def relu(x):
    res = np.zeros(x.shape)
    res[x > 0] = x[x > 0]
    return res


def drelu(x):
    res = np.zeros(x.shape)
    res[x > 0] = 1
    return res

def mse(y_pred, y):
    y = y.reshape((y.shape[0], 1))
    err = np.mean(np.square(y_pred - y))
    return err


def dmse(y_pred, y):
    y = y.reshape((y.shape[0], 1))
    n = y.shape[0]
    return y_pred - y


def cross_entropy(X, y):
    """
    X is the output from fully connected layer (num_examples x num_classes)
    y is labels (num_examples x 1)
    	Note that y is not one-hot encoded vector.
    	It can be computed as y.argmax(axis=1) from one-hot encoded vectors of labels if required.
    """
    y = y.reshape((y.shape[0], 1))
    m = y.shape[0]
    p = softmax(X.T)
    # We use multidimensional array indexing to extract
    # softmax probability of the correct label for each sample.
    # Refer to https://docs.scipy.org/doc/numpy/user/basics.indexing.html#indexing-multi-dimensional-arrays for understanding multidimensional array indexing.
    log_likelihood = -np.log(p[range(m), y])
    loss = np.sum(log_likelihood) / m
    return loss


def delta_cross_entropy(X, y):
    """
    X is the output from fully connected layer (num_examples x num_classes)
    y is labels (num_examples x 1)
    	Note that y is not one-hot encoded vector.
    	It can be computed as y.argmax(axis=1) from one-hot encoded vectors of labels if required.
    """
    y = y.reshape((y.shape[0], 1))
    m = y.shape[0]
    grad = softmax(X.T)
    grad[range(m), y] -= 1
    grad = grad / m
    return grad.T
