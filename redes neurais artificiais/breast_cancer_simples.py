import pandas as pd

atributos = pd.read_csv('entradas_breast.csv')
doenca = pd.read_csv('saidas_breast.csv')

from sklearn.model_selection import train_test_split
atributos_treinamento, atributos_teste, doenca_treinamento, doenca_teste = train_test_split(atributos,doenca, test_size=0.25)

import keras
from keras.models import Sequential
from keras.layers import Dense

classificador = Sequential()
classificador.add(Dense(units = 16, activation = 'relu', 
                        kernel_initializer = 'random_uniform', input_dim = 30))
classificador.add(Dense(units = 16, activation = 'relu',                      #duas camadas ocultas
                        kernel_initializer = 'random_uniform'))
classificador.add(Dense(units = 1, activation = 'sigmoid'))

otimizador = keras.optimizers.Adam(lr = 0.001, decay=0.0001, clipvalue = 0.5) #otimizador personalizado
classificador.compile(optimizer= otimizador,loss = 'binary_crossentropy',     #compilacao
                      metrics=['binary_accuracy'])
classificador.fit(atributos_treinamento, doenca_treinamento,                 #treinamento
                  batch_size = 10, epochs = 100)

pesos0 = classificador.layers[0].get_weights()
print(pesos0)
print(len(pesos0))
pesos1 = classificador.layers[1].get_weights()
pesos2 = classificador.layers[2].get_weights()

previsoes = classificador.predict(atributos_teste)                           
previsoes = (previsoes > 0.5)

#resultado
from sklearn.metrics import confusion_matrix, accuracy_score
precisao = accuracy_score(doenca_teste, previsoes)
matriz = confusion_matrix(doenca_teste, previsoes)

##outro método pelo keras
resultado = classificador.evaluate(atributos_teste, doenca_teste)