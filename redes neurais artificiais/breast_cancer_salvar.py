# -*- coding: utf-8 -*-
"""
Created on Sat May  2 02:24:11 2020

@author: mathe
"""


import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout

atributos = pd.read_csv('entradas_breast.csv')
doenca = pd.read_csv('saidas_breast.csv')

classificador = Sequential()
classificador.add(Dense(units = 8, activation = 'relu', kernel_initializer = 'random_uniform',
                        input_dim = 30))
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 8, activation = 'relu', kernel_initializer = 'random_uniform',))
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 1, activation = 'sigmoid'))
classificador.compile(optimizer = 'adam',loss = 'binary_crossentropy', metrics = ['binary_accuracy'])
classificador.fit(atributos, doenca, batch_size = 10, epochs = 100)

classificador_json = classificador.to_json()
with open('classificador_breast.json', 'w') as json_file:       #salvando a estrutura
    json_file.write(classificador_json)
classificador.save_weights('classificador_breast.h5')           #salvando estrutura