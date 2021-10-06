# -*- coding: utf-8 -*-
"""
Created on Sun Oct  3 22:22:20 2021

@author: Leonid
"""




import numpy as np
import pandas as pd
from my_date_transform import str2date
from my_date_transform import datetime2date



from sklearn.base import BaseEstimator, TransformerMixin
import matplotlib.pyplot as plt
from datetime import datetime



class get_period(BaseEstimator, TransformerMixin):
    def __init__(self, max_date):
        
        if type(max_date)==str:
            max_date=datetime.strptime(max_date, '%Y-%m-%d')
            max_date=max_date.date()
        self.max_date=max_date
        return
    def fit(self, X, y = None):
        
        return self
    
    def transform(self, X):
        #на вход подаем историю продаж из базы данных по конкретному товару
        #на выходе агрегирование продаж по датам и преобразование 
        #к временному ряду
        df5=X.copy()
        df5['created_at']=df5['created_at'].apply(str)
        df5['datetime']=df5['created_at'].apply(str2date)
        df5['date']=df5['datetime'].apply(datetime2date)
        df5=df5[['quantity','date']]
        df5=df5.groupby(['date'])['quantity'].sum()
        df5=df5.reset_index()
        # T=pd.date_range(df5['date'].min(),df5['date'].max(), freq='D')
        T=pd.date_range(df5['date'].min(), self.max_date, freq='D')
        T=pd.DataFrame(T, columns=['date'])
        T['date']=T['date'].apply(datetime2date)
        
        df5=pd.merge(T, df5, how='left')
        df5['quantity']=df5['quantity'].fillna(0)
        df5['date']=(df5['date']-df5['date'].min()).dt.days
        
        return df5

if __name__ == '__main__':
    df4=pd.read_csv('df4.csv')
    trans=get_period()
    trans=trans.fit(df4)
    df5=trans.transform(df4)
    plt.close('all')
    plt.figure()
    plt.plot(df5['date'], df5['quantity'], 'o-')
