import numpy as np


def MSE(y_pred,y_true):
    return np.mean((y_pred - y_true)**2)
