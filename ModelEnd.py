# -*- coding: utf-8 -*-
"""

21/5000
model design
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor as KNR

from sklearn.linear_model import LinearRegression as LR
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.ensemble import GradientBoostingRegressor as GBR
from sklearn.neural_network import MLPRegressor

from ConCorCoeff import concordance_correlation_coefficient as CCC
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler

def model(X_train, y_train, X_test=np.array([]), y_test=np.array([]), 
          method="LR"):
    #X_train inputs of model for training
    #X_test inputs of model fortesting
    #y_train -outputs for Xtrain
    #y_test - outputs fo X_test
    #method of model design. Default method is linear regression
    
    if method=="LR":
        lr=LR()
    elif method=="Ridge":
        lr=Ridge()
    elif method=="Lasso":
        lr=Lasso()
    elif method=="MLPRegressor":
        lr=MLPRegressor()
    elif method=="SVR":
        lr=SVR()
    elif method=="KNR":
        lr=KNR()
    elif method=="RFR":
        lr=RFR()
    elif method=="GBR":
        lr=GBR()
    else:
        print("unknown method")
        return False
    
        
        
#    lr = MLPRegressor( hidden_layer_sizes=[5], activation ="relu")
#    lr = MLPRegressor()
#    lr=SVR()
#    lr=KNR()
    #
#    lr=Ridge(alpha=alpha.x)
#    lr=Ridge()
#    lr=Lasso(alpha=0.001)
  
#    lr=Lasso()

#    lr=RFR(n_estimators=5, max_features=2, max_depth=2, random_state=2)
#    lr=RFR()

#    lr=GBR()
    
    lr=lr.fit(X_train, y_train[:,0])
    y_mod_train=lr.predict(X_train)
    c_train=CCC(y_train, y_mod_train[:, np.newaxis])
    
    c_test=-1
    if len(y_test)>0:
        y_mod_test=lr.predict(X_test)
        c_test=CCC(y_test, y_mod_test[:, np.newaxis])
    
    return (lr, c_train, c_test)



    

    
    
    
    