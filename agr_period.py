# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 00:19:43 2021
трансформатор данных при погнозировани спроса при управлении запасом
Исходный временной ряд агрегируем по периоду принятия решения
Период принятия решения задается в днях и равент разнице между точкой заказа
и точкой поставки СЛЕДУЮЩЕГО заказа. 

После данные передаются в get_glubina, где формируется выборка методом погружения:
Xt=F(t-1, Xt-1, Xt-2, ... Xt-k)

@author: Leonid
"""
import numpy as np
import pandas as pd

from sklearn.decomposition import LatentDirichletAllocation

from sklearn.base import BaseEstimator, TransformerMixin
import matplotlib.pyplot as plt


class agr_period(BaseEstimator, TransformerMixin):
    def __init__(self, period=11  ):
        self.period = period
        return
    def fit(self, X, y = None):
        return self
    
    def transform(self, X):
        #X - df формата:
        #df['date'] - столбец с датами (с точностью до дня)
        #df['quantity'] - количество проданное за соответствующий день
        #скользящим окном агрегируем  sum(df['quantity'].iloc[i:self.period+i ])
        #и привязываем к дате i
        rez=[]
        for i in range(len(X)-self.period):
            rez.append([X['date'].iloc[i], X['quantity'].iloc[i:i+self.period].sum()])
        rez=pd.DataFrame(rez, columns=['date','quantity' ])
        
        return rez
        
        
if __name__ == '__main__':
    df5=pd.read_csv('df5.csv')
    trans=agr_period(11)
    trans=trans.fit(df5)
    df6=trans.transform(df5)
    plt.close('all')
    plt.figure()
    plt.plot(df5['date'], df5['quantity'], 'o-')
    plt.plot(df6['date'], df6['quantity'], 'o-')
    df6.to_csv('df6.csv')

            
    
