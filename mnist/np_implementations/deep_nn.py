import numpy as np
from sklearn.metrics import mean_squared_error

class LinearLayer():
    def __init__(self,node_size,input_dims=None,activation='sigmoid'):
        self.node_size = node_size

        self.activation = activation

        self.weights = None
        self.bias = None
        self.Z = None
        self.A = None

        self.prev_A = None
        
        if input_dims:
            self.generate_weights(input_dims)

        self.grads = {}
        
    def generate_weights(self, input_dims):
        self.weights = np.random.randn(self.node_size, input_dims) * 0.01
        self.bias = np.zeros((self.node_size,1))

    def activate(self, x):
        if self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        
    def update(self):
        self.weights = self.weights - self.lr * self.grads['W']
        self.bias = self.bias - self.lr * self.grads['b']


    def forward(self, prev_A):
        # input: (input_dims, batch_size)

        if self.weights is None:
            self.generate_weights(prev_A.shape[0])

        # print(np.dot(self.weights, input))


        # linear function (Z = Wx + b)
        # takes in A of previous layer, shape: (input_dims, batch_size)
        # outputs Z of shape: (node_size, batch_size)
        # print(self.weights)
        self.Z = np.dot(self.weights, prev_A) + self.bias

        # activation function
        a = self.activate(self.Z)

        # cache for backpropagation
        self.prev_A = prev_A
        self.A = a
        
    def backward(self, grad_A):

        # grad_A: gradient of loss function wrt A of this layer

        # grad_Z, shape: (node_size, batch_size)
        grad_Z = grad_A * (self.activate(self.Z) * (1 - self.activate(self.Z) ))

        # grad_W, shape: (node_size, input_dims)
        grad_W = 1/self.prev_A.shape[1] * (np.dot(grad_Z, self.prev_A.T))

        # grad_b, shape: (node_size, 1)
        grad_b = 1/self.prev_A.shape[1] * (np.sum(grad_Z, axis=1, keepdims=True))

        # grad_prev_A, shape: (input_dims, batch_size)
        grad_prev_A = np.dot(self.weights.T, grad_Z)
        
        self.grads = {'W': grad_W, 'b': grad_b, 'prev_A': grad_prev_A,'Z': grad_Z}

        self.update()


# TODO: factor the learning rate
class BaseNN():
    def __init__(self, max_iter = 1000000,lr = 0.01, tol=0.01):

        self.max_iter = max_iter
        self.tol = tol
        self.lr = lr

        self._layers = []
        
    def forward(self,A):
        pass
    
    def backward(self,grad_A):
        pass

        
    def fit(self, X: np.ndarray = None ,y : np.ndarray = None):
        
        X = X.T

        m = X.shape[1]

        _iter = 0
        
        while True: 

            #declare A[0]
            
            A = self.forward(X)

            grad_A = -(y/A) + (1-y)/(1-A)
            
            if (_iter % 100) == 0:
                print(mean_squared_error(y,A.ravel()))
            
            self.backward(grad_A)
            
            
            if _iter == self.max_iter or mean_squared_error(y,A.ravel()) < self.tol:
                print(mean_squared_error(y,A.ravel()))
                break
            

            _iter += 1
            
        print(self)


    def predict(self,X):
        
        X = X.T
        A = self.forward(X)

        pred = np.where(A>0.5,1,0)

        return pred.ravel()



# class DNN_backup():
#     def __init__(self, X,y, max_iter = 100, tol=0.01, input_dims=None):
#         linear1 = LinearLayer(4, input_dims,activation='sigmoid')
#         # linear2 = LinearLayer(2,activation='sigmoid')
#         linear3 = LinearLayer(1,activation='sigmoid')
#         self.layers = [linear1,linear3]
#         self.max_iter = max_iter
#         self.tol = tol

#         self.fit(X,y)


#     def fit(self,X,y):
        
#         X = X.T

#         m = X.shape[1]

#         _iter = 0
        
#         while True: 

#             #declare A[0]
#             A = X
#             for layer in self.layers:
#                 #forward pass
#                 layer.forward(A)

#                 # update A for the next layer
#                 A = layer.A

#             grad_A = -(y/A) + (1-y)/(1-A)
            
#             #plot the cost function
#             for layer in reversed(self.layers):
#                 layer.backward(grad_A)
#                 layer.update()
#                 grad_A = layer.grads['prev_A']

#             if _iter == self.max_iter or loss.MSE(A,y) < self.tol:
#                 break

#         print(self)


#     def predict(self,X):
#         A = X.T
#         for layer in self.layers:
#             layer.forward(A)
#             A = layer.A

#         pred = np.where(A>0.5,1,0)

#         return pred.ravel()
