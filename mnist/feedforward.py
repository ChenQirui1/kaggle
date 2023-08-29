import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from np_implementations.nn import BaseNN
from np_implementations.layers import Linear, Sigmoid

class TestNN(BaseNN):
    def __init__(self,lr):
        super().__init__()
        
        self.lr = lr
        
        self.fc1 = Linear(4,784,lr=0.05)
        self.rel1 = TanH()
        self.fc2 = Linear(1,4,lr=0.05)
        self.sig2 = Sigmoid()
        

        
    def forward(self,X):
        X = self.fc1.forward(X)
        X = self.rel1.forward(X)
        X = self.fc2.forward(X)
        X = self.sig2.forward(X)
        
        return X
    
    
    def backward(self, grad):
        grad = self.sig2.backward(grad)
        grad = self.fc2.backward(grad)
        grad = self.rel1.backward(grad)
        grad = self.fc1.backward(grad)
        
        return grad

    
#process dataset
df = pd.read_csv("mnist/mnist/train.csv")
binary_df = df.loc[df['label'].isin([0,1])]
binary_df['label']

X = binary_df.drop('label', axis=1).values
y = binary_df['label'].values
#train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print('Shape of X: ', X.shape)
print('Shape of y: ', y.shape)

#normalise the data to 255
X_train = X_train / 255
# X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))

#initialise model
model = TestNN(lr=0.04)
# print(model.__repr__)
model.fit(X_train,y_train)


y_pred = model.predict(X_test/255)
print(confusion_matrix(y_test, y_pred, labels=[0, 1]))