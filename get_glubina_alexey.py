# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 00:19:43 2021
Данные подготавливаются по модели Алексея. Суть в том, что в прошлом выбирается
период n1 за который находим среднее значение потребления за день. 
аппроксимируем последующие дни этим значением. Находим суммарное потребление
за следующие количество дней равное периоду управления n2. Период управление - количество
дней между первым заказом и второй поставкой

формат входных данных: таблица из двух столбцов дата, количество
Формат выходных данных: таблица из трех столбцов дата, среднее количество за n1 предшествующих дней 
                                                        сумма за n2 последующих дней

Например на входе

t_i-n1-1 x_i-n1-1
...
t_i     x_i
t_i+1   x_i+1
...
t_i+n2-1 x_i+n2

на выходе
t_i среднее_значение(x_i-n1-1, x_i-n1, ...x_i-1) сумма(x_i, x_i+1, ...x_i+n2)
@author: Leonid
"""
import numpy as np
import pandas as pd

from sklearn.decomposition import LatentDirichletAllocation

from sklearn.base import BaseEstimator, TransformerMixin
import matplotlib.pyplot as plt


class get_glubina_alexey(BaseEstimator, TransformerMixin):
    def __init__(self, n1=5, n2=2 ):
        self.n1 = n1
        self.n2=n2
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
        n1=self.n1
        n2=self.n2
        if   (self.n1+self.n2)>=len(X):
            return False
        rez=[]
        for i in range(n1+1,len(X)-n2):
            rez.append([X['date'].iloc[i], X['quantity'].iloc[i-n1-1:i].mean(),
                                                X['quantity'].iloc[i:i+n2+1].sum() ])
        rez=pd.DataFrame(rez, columns=['date', 'in', 'out'] )
        
        return rez
        
        
if __name__ == '__main__':
    df5=pd.read_csv('df5.csv')
    trans=get_glubina_alexey(n1=5, n2=2)
    trans=trans.fit(df5)
    df7=trans.transform(df5)
   

            
    
