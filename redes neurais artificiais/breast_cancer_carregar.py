# -*- coding: utf-8 -*-
"""
Created on Sat May  2 02:36:02 2020

@author: mathe
"""


import numpy as np
import pandas as pd
from keras.models import model_from_json

arquivo = open('classificador_breast.json' , 'r')
estrutura_rede = arquivo.read()
arquivo.close()

classificador = model_from_json(estrutura_rede)
classificador.load_weights('classificador_breast.h5')

novo = np.array([[15.80, 8.34, 118, 900, 0.10, 0.26, 0.08, 0.134, 0.178, 0.20, 
                  0.05, 1098, 0.87, 4500, 145.2, 0.005, 0.04, 0.05, 0.015, 0.03,
                  0.007, 23.15, 16.64, 178.5 ,2018, 0.14, 0.185, 0.84, 158, 0.363]])

previsao = classificador.predict(novo)
previsao = previsao > 0.5

atributos = pd.read_csv('entradas_breast.csv')
doenca = pd.read_csv('saidas_breast.csv')

classificador.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['binary_accuracy'])
resultado = classificador.evaluate(atributos, doenca)