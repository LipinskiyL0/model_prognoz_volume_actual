# -*- coding: utf-8 -*-
"""
Created on Sat Jul  3 00:15:35 2021

@author: Leonid
"""
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import r2_score

class model_mean(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
        return
    def fit(self, X, labels = None ):
        pass
        return self
    def predict(self, X):
        X1=X.copy()
        if 'date' in X1.columns:
            del X1['date']
        return X1.mean(axis=1)
    def score(self, X, y=None):
        y_pred=self.predict(X)
        return r2_score(y_pred,y )
    
        
         