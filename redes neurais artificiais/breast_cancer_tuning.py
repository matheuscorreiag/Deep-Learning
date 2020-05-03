# -*- coding: utf-8 -*-
"""
Created on Fri May  1 02:01:04 2020

@author: mathe
"""


import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

atributos = pd.read_csv('entradas_breast.csv')
doenca = pd.read_csv('saidas_breast.csv')

def criarRede(optimizer, loss,kernel_initializer, activation, neurons):
    
    classificador = Sequential()
    classificador.add(Dense(units = 32, activation = activation, 
                            kernel_initializer = kernel_initializer, input_dim = 30))
    classificador.add(Dropout(0.2))
    classificador.add(Dense(units = neurons, activation = activation, kernel_initializer = kernel_initializer))
    classificador.add(Dropout(0.2))
    classificador.add(Dense(units = neurons, activation = activation, kernel_initializer = kernel_initializer))
    classificador.add(Dropout(0.2))
    classificador.add(Dense(units = 1, activation = 'sigmoid'))

    classificador.compile(optimizer = optimizer, loss = loss, metrics = ['binary_crossentropy'])
    
    return classificador

classificador = KerasClassifier(build_fn = criarRede)
parametros = {'batch_size':[10],
              'epochs': [50],
              'optimizer': ['adam'],
              'loss': ['binary_crossentropy'],
              'kernel_initializer': ['random_uniform', 'normal'],
              'activation': ['relu'],
              'neurons':[8,32]}

grid_search = GridSearchCV(estimator = classificador,
                           param_grid = parametros,
                           scoring = 'accuracy',
                           cv = 5)

grid_search = grid_search.fit(atributos, doenca)
melhores_parametros = grid_search.best_params_
melhor_precisao = grid_search.best_score_