# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 00:46:54 2020

@author: Леонид

В файле реализуется модель, которая будет состоять из стандартных моделей 
библиотеки sklearn, но с класстеризацией
на каждый кластер будет своя модель. При расчете автоматически определяется 
к какому кластеру относится объект и подгружается соответствующая модель

n_claeters - количество кластеров на которое разбивается пространство
base_model - тип модели на основе которых будут строится модели по кластерам
models - словарь моделей 
kmeans - классификатор
f_minmax - флаг определяет берем минимум=min, всех кривых, максимум=max,
            или оставляем как есть=""
"""
from sklearn.cluster import KMeans
from ModelEnd import model
import numpy as np
class col_model:
    def __init__(self, n_clasters=2, base_model="LR"):
        self.n_clasters=n_clasters
        self.base_model=base_model
        self.f_minmax="min"
    def fit(self, X, y):
        #проводим кластеризацию и обучение модели
        # строим модель кластеризации
        
        kmeans = KMeans(n_clusters=self.n_clasters)
        kmeans.fit(X)
        
        labels=kmeans.predict(X)
        self.kmeans=kmeans
        cl=list(set(labels))
        cl.sort()
        #по полученным кластерам строим модель
        self.models={}
        for c in cl:
            ind=labels==c
            X1=X[ind, :]
            y1=y[ind]
            
            rez=model(X_train=X1, y_train=y1[:,np.newaxis], method=self.base_model)
            self.models[c]=rez[0]
        if self.n_clasters==1:
            return self
        #Если тренд линейный определяем параметры сглаживания тренда
        if ((self.base_model=="LR")|(self.base_model=="Ridge")|(self.base_model=="Lasso")):
            #линейные модели
            #определяем модель с наименьшей ценой и по ней определяем
            #если коэффициент отрицательный то тренд вогнутый вниз 
            #берем максимум
            #если хоть один коэффициент положительный то строим одну модель  
            #если все коэффициенты отрицательные, то сглаиваем тренд через максимум моделей
            #если другая фигня 
            #выравнивания не делаем
            cent=np.array(kmeans.cluster_centers_)
            ind=np.argmin(cent)
            ind2=np.argmax(cent)
            coef=[]
            for i in self.models:
                coef.append(self.models[i].coef_[0])
            coef=np.array(coef)
            
            if np.sum(coef>0)>0:
                self.f_minmax=""
                self.n_clasters=1
                self=self.fit(X,y)
                
            else:
                if self.models[ind].coef_[0]> self.models[ind2].coef_[0]:
                    self.f_minmax="min"
                else:
                    self.f_minmax="max"
            #проверяем, если тренд улетает в отрицательную область
            #приводим к одной модели
            y_try=self.predict( X, negativ_num=True)
            if np.min(y_try)<0:
                self.f_minmax=""
                self.n_clasters=1
                self=self.fit(X,y)
                
        else:
            self.f_minmax=""
        return self
    
    def predict(self, X, negativ_num=False):
        
        
        y_pred=np.zeros(len(X[:,0]))
        labels=self.kmeans.predict(X)
        
        cl=list(set(labels))
        
        #по полученным кластерам строим модель
        if self.f_minmax=="":
            #рассчитываем тренды как есть
            for c in cl:
                ind=labels==c
                X1=X[ind, :]
                
                y_pred[ind]=self.models[c].predict(X1)
        elif self.f_minmax=="min":
            #Рассчитываем тренды по минимуму
            y_temp=np.zeros([len(X[:,0]), len(self.models)])
            i=0
            for m in self.models:
                y_temp[:, m]=self.models[m].predict(X)
                i+=1
            y_pred=np.min(y_temp, axis=1)
        elif self.f_minmax=="max":
            #рассчитываем тренды по максимуму
            y_temp=np.zeros([len(X[:,0]), len(self.models)])
            i=0
            for m in self.models:
                y_temp[:, m]=self.models[m].predict(X)
                i+=1
            y_pred=np.max(y_temp, axis=1)
        else:
            return False
            
        if negativ_num==False:
            y_pred[y_pred<0]=0
        
        return y_pred
            
        
            
        
        
        
        
        
        

