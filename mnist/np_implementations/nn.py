import numpy as np
from sklearn.metrics import mean_squared_error


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
