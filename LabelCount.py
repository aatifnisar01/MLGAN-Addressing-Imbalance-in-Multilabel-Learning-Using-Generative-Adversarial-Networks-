# -*- coding: utf-8 -*-
"""
Created on Wed May 11 23:53:02 2022

@author: Aatif
"""

import pandas as pd
df=pd.read_csv('mirflickr.csv')

X = df.iloc[:, :150]
y = df.iloc[:, 150:]

xx = y.sum(axis=1)


mean = xx.mean()