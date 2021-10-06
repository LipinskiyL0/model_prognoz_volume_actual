# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 00:19:43 2021
трансформатор данных при погнозировани спроса при управлении запасом

Является новой версией файла get_period.py
В отличии от предудущей версии тут происходит коррекция данных на основании информации о скидках
Рассчитывается модель эластичности, которая вычсляет тренд изменения объемов с 
изменением цены. На основании этой модели происходит коррекция объемов в чеках, 
где цена указана со скидкой. пропорционально тому, как сильно цена отличается от базовой

Алгоритм: 
    fit:
    1 Берется исходная выборка с полями: 
    'date',  'total','quantity'. Вычисляется фактическая
    цена продажи 'price1'='total'/'quantity'. Агрегируются данные по:
    'date', с усреднением 'price1' и суммированием 'quantity'.
    Предполагается, что выборка собрана по одному 'product_id' и одному 'warehouse_id'
    
    2. Строится модель эластичности на полученных данных.
    
    predict:
    1. Берется исходная выборка. Для каждого чека, в котором цена со скидкой стоит
    пустой заполняем основной ценой. В итоге для таких чеков разница, между ценой со скидкой и 
    ценой без скидки = 0. 
    2. Для каждого чека вычисляем коэффициент понижения из пропорции: 
        объем при цене без скидки по модели так относится к объему при цене со скидкой по модели, 
        как объем при цене без скидки в чеке к объему при цене со скидкой в чеке.
        от сюда,
        объем при цене без скидки в чеке= (Объем при цене со скидкой в чеке)*(объем при цене без скидки по модели)/(объем при цене со скидкой по модели)
        т.е. (объем при цене без скидки по модели)/(объем при цене со скидкой по модели) есть коэффициент пропорциональности
        Если (объем при цене без скидки в чеке) получается отрицательным или большим (Объем при цене со скидкой в чеке),
        то устанавливаем (Объем при цене со скидкой в чеке).
    3. данные группируются по дате с суммированием рассчитанных объемов продаж. Затем 
       преобразуются во временной ряд и передаются в agr_period где продажи агрегируются за период. 
    
Исходные данные это статистика продаж по конкретному товару. Ключевыми являются:
date - дата продажи
quantity - объем продаж
total - стоимость
 


@author: Leonid
"""
import numpy as np
import pandas as pd
from my_date_transform import str2date
from my_date_transform import datetime2date

from sklearn.decomposition import LatentDirichletAllocation

from sklearn.base import BaseEstimator, TransformerMixin
import matplotlib.pyplot as plt
from datetime import datetime
from col_model import col_model

class get_period(BaseEstimator, TransformerMixin):
    def __init__(self, max_date):
        
        if type(max_date)==str:
            max_date=datetime.strptime(max_date, '%Y-%m-%d')
            max_date=max_date.date()
        self.max_date=max_date
        return
    def fit(self, X, y = None):
        
        df5=X.copy()
        df5['price1']=df5['total']/df5['quantity']
        df5['datetime']=df5['created_at'].apply(str2date)
        df5['date']=df5['datetime'].apply(datetime2date)
        # X1=X[['date', 'price1','quantity']]
        df5=df5.groupby(['date','product_id', 'warehouse_id']).aggregate({'price1': 'mean','quantity': 'sum'})
        df5=df5.reset_index()
        
        
        if len(df5)<30:
            lr=col_model( n_clasters=1, base_model="LR")
        else:
            lr=col_model( n_clasters=2, base_model="LR")
        lr=lr.fit(df5[['price1']].values, df5["quantity"].values)
        self.lr=lr
        del df5
        return self
    
    def transform(self, X):
        #на вход подаем историю продаж из базы данных по конкретному товару
        #на выходе агрегирование продаж по датам и преобразование 
        #к временному ряду
        df5=X.copy()
        df5['created_at']=df5['created_at'].apply(str)
        df5['datetime']=df5['created_at'].apply(str2date)
        df5['date']=df5['datetime'].apply(datetime2date)
        ind=df5['discounted_price'].isnull()
        df5.loc[ind, 'discounted_price']=df5.loc[ind, 'price']
        df5['k']=self.lr.predict(df5[['price']].values)/self.lr.predict(df5[['discounted_price']].values)
        ind=(df5['k']<0) | (df5['k']>1)
        df5.loc[ind, 'k']=1
        df5["quantity1"]=df5["quantity"]*df5["k"]
        df5=df5[['quantity1','date']]
        df5.columns=['quantity','date']
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
    trans=get_period('2021-06-30')
    trans=trans.fit(df4)
    df5=trans.transform(df4)
    plt.close('all')
    plt.figure()
    plt.plot(df5['date'], df5['quantity'], 'o-')

    

            
    
