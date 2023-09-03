import numpy as np
from sklearn.metrics import mean_squared_error

# class CostFunc(y_true,y_pred,reg="l2",lambd=0.2):

#     def __init__(self,y_true,y_pred,reg,lambd,w) -> None:
        
#         self.lambd = lambd
#         self.reg = reg
        
#         self._run()


#     def _run(self,y_true,y_pred,lambd w):

#         if self.reg == "l2":
#             norm = (lambd/len(y_true)) * (w * w)
        
        
#         mean_squared_error(y_true,y_pred) + _l2(lambd)
        
#         return mean_squared_error(y_true,y_pred)


def Cost(y_true,y_pred,norm,lambd,w):

    if norm == "l2":
        decay = (lambd/len(y_true)) * (w * w)
    else:
        decay = None

    return mean_squared_error(y_true,y_pred) + decay
