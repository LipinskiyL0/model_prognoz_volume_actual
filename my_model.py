# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 00:32:11 2021

@author: Leonid
"""


from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor as KNR

from sklearn.linear_model import LinearRegression as LR
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.ensemble import GradientBoostingRegressor as GBR
from sklearn.neural_network import MLPRegressor
from model_mean import model_mean
from model_exp_mean import model_exp_mean
from sklearn.metrics import r2_score


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class my_model(BaseEstimator, TransformerMixin):
    def __init__(self,name_model='LR',n_test=5, alpha=0.5 ):
        self.name_model=name_model
        self.n_test=n_test
        self.alpha=alpha
        return 
    
    
    def fit(self, X, labels = None, ):
        n=len(X)
        X_in=X.iloc[:n-self.n_test, :-1]
        X_out=X.iloc[:n-self.n_test, -1]
        
        if self.name_model=="LR":
            lr=LR()
        elif self.name_model=="Ridge":
            lr=Ridge()
        elif self.name_model=="Lasso":
            lr=Lasso()
        elif self.name_model=="MLPRegressor":
            lr=MLPRegressor()
        elif self.name_model=="SVR":
            lr=SVR()
        elif self.name_model=="KNR":
            lr=KNR()
        elif self.name_model=="RFR":
            lr=RFR()
        elif self.name_model=="GBR":
            lr=GBR()
        elif self.name_model=="GBR":
            lr=GBR()
        elif self.name_model=="MM":
            lr=model_mean()
        elif self.name_model=="MeM":
            lr=model_exp_mean(alpha=self.alpha)
        else:
            print("unknown method")
            return False
        if len(X_out)==0:
            print('Недостаточно данных для обучения')
            return False
        lr=lr.fit(X_in,X_out)
        self.model=lr
        self.n_in=X_in.shape[1]
        # print( 'размерность', X_in.shape[1])
        return self
    
    def predict(self, X):
        
        flag=0
        if X.shape[1]==self.n_in:
             X_in=X
            
        elif X.shape[1]-1==self.n_in:
            xx=X.iloc[-1, :]
            xx=pd.DataFrame(xx)
            xx=xx.T
            ind=xx.index[0]
            xx.loc[ind,'date']= xx.loc[ind,'date']+1
            for i in range(self.n_in-1):
                xx.iloc[0,i+1]=xx.iloc[0,i+2]
            X1=pd.concat([X, xx], axis=0)
            X_in=X1.iloc[:, :-1]
            self.X_out=X.iloc[:, -1]
            
        else:
            return False
            
            
        y=self.model.predict(X_in)
        return y
    
    def score(self, X, y=None):
        X_in=X.iloc[:, :-1]
        X_out=X.iloc[:, -1]
        
        X_out_pred=self.model.predict(X_in)
        return r2_score(X_out,X_out_pred )
        
        
    

