import numpy as np
import math
from sklearn.metrics import mean_squared_error

#
def minibgd(X,y,model,batch_size):

    #mini batch gradient descent
    X = np.split(X,batch_size)

    y = np.split(y,batch_size)

    epochs = y.shape[0] // batch_size

    #forward prop
    for epoch in enumerate(epochs):
        A = model.forward(X[epoch])

        grad_A = -(y[epoch]/A) + (1-y[epoch])/(1-A)

        # mean_squared_error(y,A.ravel())

        model.backward(grad_A)




if __name__ == "__main__":
    print(math.ceil(13/2))