import numpy as np
from np_implementations.deep_nn import BaseNN

class Linear(BaseNN):
    def __init__(self,node_size,input_dims=None,lr=None):
        super().__init__()
        
        self.node_size = node_size
        
        self.weights = None
        self.bias = None
        self.Z = None
        self.prev_A = None
        
        if lr != None:
            self.lr = lr
            
        if input_dims:
            self.generate_weights(input_dims)

        self.grads = {}
                    
        
    def generate_weights(self, input_dims):
        self.weights = np.random.randn(self.node_size, input_dims) * 0.01
        self.bias = np.zeros((self.node_size,1))

        
    def update(self):
        self.weights = self.weights - self.lr * self.grads['W']
        self.bias = self.bias - self.lr * self.grads['b']


    def forward(self, prev_A):
        # input: (input_dims, batch_size)

        if self.weights is None:
            self.generate_weights(prev_A.shape[0])

        # linear function (Z = Wx + b)
        # takes in A of previous layer, shape: (input_dims, batch_size)
        # outputs Z of shape: (node_size, batch_size)
        # print(self.weights)
        self.Z = np.dot(self.weights, prev_A) + self.bias

        # cache for backpropagation
        self.prev_A = prev_A
        
        return self.Z
    

    def backward(self, grad_Z):

        # grad_W, shape: (node_size, input_dims)
        grad_W = 1/self.prev_A.shape[1] * (np.dot(grad_Z, self.prev_A.T))

        # grad_b, shape: (node_size, 1)
        grad_b = 1/self.prev_A.shape[1] * (np.sum(grad_Z, axis=1, keepdims=True))

        # grad_prev_A, shape: (input_dims, batch_size)
        grad_prev_A = np.dot(self.weights.T, grad_Z)

        self.grads = {'W': grad_W, 'b': grad_b, 'prev_A': grad_prev_A,'Z': grad_Z}
        
        self.update()
        
        return grad_prev_A

# TODO: nest other activations 
class Sigmoid():
    
    def __init__(self):
        self.A = None
    
    #cache/return A
    def forward(self, Z):
        
        self.A = 1 / (1 + np.exp(-Z))
        
        return self.A
    
    #return grad_Z
    def backward(self, grad_A):
        
        # grad_A: gradient of loss function wrt A of this layer
        # grad_Z, shape: (node_size, batch_size)
        grad_Z = grad_A * (self.A * (1 - self.A))
        
        return grad_Z