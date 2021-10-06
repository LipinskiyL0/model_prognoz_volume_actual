# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 17:28:36 2021

Общая схема работы:
    get_prognoz.get_prognoz(max_date) - принимает max_date - максимальная дата периода обучения 
                                        max_date - параметр, который нужен для корректного рассчета
                                        в тех случаях, когда за последние дни не было продаж и объемы по нулям. 
                                        Что бы эти нули появились в выборке нужно дату определять не по выборке, 
                                        а указать в качестве параметра
                                        get_prognoz.get_prognoz(max_date) - подключаемся к выборке и извлекаем по очереди 
                                        выборку по конкретному product_id и warehouse_id (df4). Формируем выборку под прогноз 
                                        и вызываем my_pipeline_linear.predict_model_linear(df4, period, max_date)
    
    my_pipeline_linear.predict_model_linear(df4, period, max_date) - вычисляет прогнозное значение. df4 выборка по конкретному товару
                                        с конкретного склада. period - период управления=периоду на который делаем заказ, измеряется в днях.
                                        max_date - максимальная дата периода обучения. Для построения прогноза последовательно вызываются:
                                        get_period.get_period(max_date)
                                        agr_period.agr_period(period=period)
                                        get_glubina.get_glubina(n_glub=n_glub)
                                        my_model.my_model(name_model=name_model, n_test=5)
    
    get_period.get_period(max_date) - Агрегация данных о продажах посуточно
    agr_period.agr_period(period=period) - Агрегируем данные по заданному периоду управления
    get_glubina.get_glubina(n_glub=n_glub) - Разрезаем выборку в глубину
    my_model.my_model(name_model=name_model, n_test=5) - строим модель и вычисляем выход
    
    
    
    

    


@author: Leonid
"""

import psycopg2
import psycopg2.extras


import pandas as pd
import numpy as np
 
from datetime import datetime
from my_date_transform import str2date, str2date1
from my_date_transform import datetime2date
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression as LR

from my_pipeline_linear import predict_model_linear





#def get_prognoz(file_in, file_out):
def get_prognoz(max_date):
    #1. устанавливаем соединение с базой
    try:
        conn = psycopg2.connect(dbname='backend', user='foodrocket', 
                                password='5ADA2F02-667E-4E66-BEDD-B439D2EF5F12', host='localhost', port="5432")
        
        cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    except:
        print("Ошибка подключения к базе")
        return False
    
    try:
        df_in=pd.read_csv(file_in)
    except:
        print("не найден входящий файл", file_in)
        return False
    ind=((df_in['delivery_dt'].isnull())|(df_in['current_dt'].isnull())|
         (df_in['warehouse_id'].isnull())|(df_in['product_id'].isnull()))
    
    if (sum(ind)>0):
        print("Файл содержит пропуски. Пропуски будут удалены")
        print(df_in[ind])
    df_in=df_in.dropna()  
      
    df_in['current_dt']=df_in['current_dt'].apply(str2date1)
    df_in['delivery_dt']=df_in['delivery_dt'].apply(str2date1)
    df_in['period']=(df_in['delivery_dt']-df_in['current_dt']).dt.days
    
    #2. Закачиваем историю продаж где заказ оплачен
    df4 = pd.read_sql_query(
        '''SELECT 
                tab1."id",
                tab1."order_id",
                tab1."product_id",
                tab1."warehouse_id",
                tab1."quantity",
                tab1."created_at"
                
                
           FROM orders_items AS tab1 
           
           INNER JOIN orders_statuses AS tab2
           ON (tab1."order_id"=tab2."order_id") AND (tab2.kind=0) AND (tab2."status"=2)
           
                
        '''.format('orders_items'),conn)
    
    #3 извлекаем по одной позиции из file_in и отрабатываем прогноз
    result=[]
    for i in range(len(df_in)):
        ind=((df4['product_id']==df_in['product_id'].iloc[i])&
             (df4['warehouse_id']==df_in['warehouse_id'].iloc[i])&
             (df4['created_at']<=df_in['current_dt'].iloc[i])
             )
        
        
        try:
            y_pred=predict_model_linear(df4[ind], df_in['period'].iloc[i],max_date )
        except:
            y_pred=-1
        
        result.append([df_in['current_dt'].iloc[i], 
                       df_in['warehouse_id'].iloc[i], 
                       df_in['product_id'].iloc[i],
                       df_in['delivery_dt'].iloc[i],
                       y_pred])
    result=pd.DataFrame(result, columns=['current_dt','warehouse_id', 'product_id',
                                         'delivery_dt','forecast'  ])
    
        
    
    result.to_csv(file_out)    
    return True

if __name__=='__main__':
    file_in='sku_line_model.csv'
    file_out='result.csv'
    flag=get_prognoz(file_in, file_out)
    if (flag==False):
        print('Не выполнено из-за ошибки')