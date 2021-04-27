# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 16:26:14 2021

@author: Alan
"""

# Parte 1 - Pre procesado de datos

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el data set
dataset = pd.read_csv('heart.csv')

X = dataset.iloc[:, 0:13].values
y = dataset.iloc[:, 13].values
y=y.reshape((303,1))

# Dividir el data set en conjunto de entrenamiento y conjunto de testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Escalado de variables
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)



# Importar Keras y librerías adicionales
import keras
from keras.models import Sequential #Inicializar los parametros de la red neuronal
from keras.layers import Dense


classifier = Sequential()

# Añadir las capas de entrada y primera capa oculta
classifier.add(Dense(units = 6, kernel_initializer = "uniform",  
                     activation = "relu", input_dim = 13))

# Añadir la segunda capa oculta
classifier.add(Dense(units = 6, kernel_initializer = "uniform",  activation = "relu"))

# Añadir la capa de salida
classifier.add(Dense(units = 1, kernel_initializer = "uniform",  activation = "sigmoid"))

# Compilar la RNA
classifier.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])


# Ajustamos la RNA al Conjunto de Entrenamiento
classifier.fit(X_train, y_train,  batch_size = 10, epochs = 100)


# Parte 3 - Evaluar el modelo y calcular predicciones finales

# Predicción de los resultados con el Conjunto de Testing
y_pred = classifier.predict(X_test)
y_pred = (y_pred>0.5)
# Elaborar una matriz de confusión
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


(23+31)/61

