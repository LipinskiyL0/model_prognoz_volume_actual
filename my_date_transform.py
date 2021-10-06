# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 15:35:01 2021

@author: Leonid
"""
from datetime import datetime

def str2date(s):
    try:
        d=datetime.strptime(s, '%Y-%m-%d %H:%M:%S.%f')
    except:
        d=datetime.strptime(s, '%Y-%m-%d %H:%M:%S')
    return d
def str2date1(s):
    return datetime.strptime(s, '%Y-%m-%d')
def datetime2date(s):
    return s.date()