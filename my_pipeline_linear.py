# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 23:55:57 2021

@author: Leonid

Общая схема работы

 

"""



from sklearn.pipeline import Pipeline

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from sklearn.model_selection import GridSearchCV


from get_period import get_period
from get_period_elast_correction import get_period as get_period_elast
from agr_period import agr_period
from get_glubina import get_glubina



from my_model import my_model

def test_model_linear(df4, period, max_date, name='', f_elast=False, f_plot=False):
    #функция производит перебор параметров по схеме решетчатого поиска, запоминает 
    #наилучшие параметры и возыращает результат
    param_grid = {'glub__n_glub': [0, 1,2,3,4,5],
                  'my_model__name_model': ["LR","Ridge", "Lasso"  ]
                  
                    }
    fl_rez=0
    for n_glub in param_grid['glub__n_glub']:
        for name_model in param_grid['my_model__name_model']:
            if f_elast==False:
                pipe = Pipeline([("period", get_period(max_date)),
                         ("agr", agr_period(period=period)),
                         ('glub', get_glubina(n_glub=n_glub)),
                         ('lr', my_model(name_model=name_model, n_test=5))
                         ])
            else:
                pipe = Pipeline([("period", get_period_elast(max_date)),
                         ("agr", agr_period(period=period)),
                         ('glub', get_glubina(n_glub=n_glub)),
                         ('lr', my_model(name_model=name_model, n_test=5))
                         ])
            pipe=pipe.fit(df4)
            rez=pipe.score(df4)
            if fl_rez==0:
                fl_rez=1
                best_model=pipe
                best_n_glub=n_glub
                best_name_model=name_model
                best_rez=rez
                
            elif best_rez<rez:
                best_model=pipe
                best_n_glub=n_glub
                best_name_model=name_model
                best_rez=rez
                
    
    pipe=best_model
    y_pred=pipe.predict(df4)
    # print(pipe.score(df4))
    if f_plot:
        plt.close('all')
        plt.figure()
        plt.plot( pipe[-1].X_out.values, 'ro-')
        plt.plot(y_pred, 'bo-')
        plt.legend(['эталон', 'модель'])
        plt.title('Линейная модель\n {0}\n score={1}'.format(name, np.round(pipe.score(df4), 4)), fontsize=10)
        plt.savefig('{0} Линейная модель.png'.format(name))
    
    return {'best_n_glub':best_n_glub, 'best_name_model':best_name_model, 'best_rez':best_rez}

def predict_model_linear(df4, period, max_date):
    #функция производит перебор параметров по схеме решетчатого поиска, запоминает 
    #наилучшие параметры и вычисляет выход с наилучшими параметрами.
    
    param_grid = {'glub__n_glub': [0, 1,2,3,4,5],
                  'my_model__name_model': ["LR","Ridge", "Lasso"  ]
                  
                    }
    fl_rez=0
    for n_glub in param_grid['glub__n_glub']:
        for name_model in param_grid['my_model__name_model']:
            pipe = Pipeline([("period", get_period(max_date)),
                     ("agr", agr_period(period=period)),
                     ('glub', get_glubina(n_glub=n_glub)),
                     ('lr', my_model(name_model=name_model, n_test=5))
                     ])
            pipe=pipe.fit(df4)
            rez=pipe.score(df4)
            if fl_rez==0:
                fl_rez=1
                best_model=pipe
                best_n_glub=n_glub
                best_name_model=name_model
                best_rez=rez
                
            elif best_rez<rez:
                best_model=pipe
                best_n_glub=n_glub
                best_name_model=name_model
                best_rez=rez
                
    
    pipe=best_model
    y_pred=pipe.predict(df4)
    if y_pred[-1]<0:
        return 0
    return  y_pred[-1]
    

if __name__ == '__main__':
    df4=pd.read_csv('history-20210917-175122.csv')
    period=4
    # name='291e3fc0-e5fe-4d64-899c-75efed1cdf71'
    name='cff942e8-7614-491a-b478-50b8d1456fbf'
    
    rez=test_model_linear(df4[df4['product_id']==name], period, '2021-09-21' ,name)
   
       
    
    print("best_n_glub={0}, best_name_model={1}, best_rez={2}".format(
        rez['best_n_glub'], 
        rez['best_name_model'], rez['best_rez']))
    rez=predict_model_linear(df4[df4['product_id']==name], period, '2021-09-21')
    
    
