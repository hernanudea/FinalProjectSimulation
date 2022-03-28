import numpy as np
import matplotlib.pyplot as plt

from scipy import stats
from scipy.spatial import distance
from scipy.stats import mode

# import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.preprocessing import normalize
from sklearn.neural_network import MLPClassifier as MLP

from mlxtend.feature_selection import SequentialFeatureSelector as SFS

# from __future__ import division

import operator
import time




db = np.loadtxt('Hernan_Velasquez_Juan_Gabriel_Velasquez_database_informe.txt',delimiter=',')
print(f"Forma de la Base de Datos: {db.shape}")

X = db[:,0:30]
Y = db[:,30]


def metrics(y_test, y_pred):
    '''
    Parametros:
      - lista (array) y_test (One-hot encoding)
      - lista (array) y_predichas (softmax vector)
    Return:
      - F1
      - Recall
      - Precision
      - Accuracy
      - Especificity
    '''
    y_test = np.asarray(y_test)
    y_pred = np.asarray(y_pred)

    print('y_test[0].shape[0]', y_test[0].shape[0])

    if y_test[0].shape[0] == 1:  # Biclase
        TP, TN, FP, FN = 0, 0, 0, 0
        for t, p in zip(y_test, y_pred):
            if t == 1:
                if p >= 0.5:  # yp==1:
                    TP += 1
                else:
                    FN += 1
            else:
                if p >= 0.5:  # yp==1:
                    FP += 1
                else:
                    TN += 1

        recall = TP / (TP + FN)
        prec = TP / (TP + FP)
        acc = (TP + TN) / (TP + FP + FN + TN)
        f1 = (2 * recall * prec) / (recall + prec)
        esp = TN / (TN + FP)

        return f1, recall, prec, acc, esp

    else:  # Multiclase
        error = 0
        for test, pred in zip(y_test, y_pred):
            t = np.argmax(test)
            p = np.argmax(pred)
            if t != p:
                error += 1

        err = error / np.shape(y_test)[0]
        acc = 1 - err

        return acc


model = Sequential()
model.add(Dense(10, activation='relu', input_dim=22))
model.add(Dense(2, activation='relu'))
model.add(Dense(3, activation='softmax'))

# Metodología de validación Cross Validation de 10 folds
Accuracy = np.zeros(10)

kf = KFold(n_splits=10, shuffle=True)
j = 0
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]

    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

    # Entrenamiento del modelo
    model.fit(X_train, y_train, epochs=50, batch_size=100)

    # Validación modelo
    y_pred = model.predict(X_test, batch_size=None, verbose=0, steps=None)

    Accuracy[j] = metrics(y_test, y_pred)
    j += 1

print("\nEficiencia en la clasificacióón: " + str(np.mean(Accuracy)) + " +/- " + str(np.std(Accuracy)))

print("\n\nTiempo total de ejecución: " + str(time.time() - tiempo_i) + " segundos.")

