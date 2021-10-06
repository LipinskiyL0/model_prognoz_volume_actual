# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 00:19:43 2021
На вход получаем временной ряд, на выходе даем выборку методом погружения
 
Xt=F(t, Xt-1, Xt-2, ... Xt-k)

X

@author: Leonid
"""
import numpy as np
import pandas as pd

from sklearn.decomposition import LatentDirichletAllocation

from sklearn.base import BaseEstimator, TransformerMixin
import matplotlib.pyplot as plt


class get_glubina(BaseEstimator, TransformerMixin):
    def __init__(self, n_glub=3 ):
        self.n_glub = n_glub
        return
    def fit(self, X, y = None):
        return self
    
    def transform(self, X):
        #X - df формата:
        #df['date'] - столбец с датами (с точностью до дня)
        #df['quantity'] - количество проданное за соответствующий день
        #делаем погружение  в текущий момент времени и в глубину истории прогнозируем
        # следующий момент времени. df['quantity'].iloc[i:self.period+i ])
        #и привязываем к дате i
       
        
        rez=[]
        for i in range(self.n_glub,len(X)):
            rez.append([X['date'].iloc[i]]+list( X['quantity'].iloc[i-self.n_glub:i+1]))
        rez=pd.DataFrame(rez, columns=['date']+['quantity'+str(i) for i in range(self.n_glub+1) ])
        # print(self.n_glub)
        return rez
        
        
if __name__ == '__main__':
    df6=pd.read_csv('df6.csv')
    trans=get_glubina(3)
    trans=trans.fit(df6)
    df7=trans.transform(df6)
   

            
    
