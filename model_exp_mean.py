# -*- coding: utf-8 -*-
"""
Created on Sat Jul  3 00:15:35 2021

@author: Leonid
"""
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import r2_score
import pandas as pd
class model_exp_mean(BaseEstimator, TransformerMixin):
    def __init__(self, alpha):
        self.alpha=alpha
        return
    def fit(self, X, labels = None ):
        pass
        return self
    def predict(self, X):
        X1=X.copy()
        
        if 'date' in X1.columns:
            del X1['date']
        
        
        Et_1=0
        y=[]
        for x in X1.iloc[:, -1]:
            Et=x*self.alpha+(1-self.alpha)*Et_1
            y.append(Et)
            Et_1=Et
        y=pd.Series(y)
        return y
    def score(self, X, y=None):
        y_pred=self.predict(X)
        return r2_score(y_pred,y )
    
        
         