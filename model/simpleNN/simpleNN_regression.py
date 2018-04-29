import numpy as np
import keras
import math
import random

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, MaxPooling1D, Convolution1D
from keras.regularizers import L1L2

from utilities import setupTrain
from utilities import model_utilities

# index 93 is ema_CALM
EMA_INDEX = 93

def regression_participant_independent(csv, trainingid, validationid, testingid, normalizerMethod, normalizer1, normalizer2, totalDays):
    X_train = []
    Y_train = []

    X_val = []
    Y_val = []

    X_test = []
    Y_test = []

    for i in range(totalDays - 1, len(csv)):
        days = setupTrain.collectDayData(csv, i, totalDays)

        if setupTrain.isSameUserAcross(days):
            x = setupTrain.transform_into_x_feature(
                days,
                True,
                "simpleNN",
                totalDays,
                "z-score",
                normalizer1,
                normalizer2
            )

            # put the x vector into the appropriate set (i.e. training, validation, testing)
            if days[0][EMA_INDEX] != '':
                X_train, Y_train, X_val, Y_val, X_test, Y_test = setupTrain.collect_train_val_test_independent(
                    False,
                    days[0][1],
                    trainingid,
                    validationid,
                    testingid,
                    X_train,
                    Y_train,
                    X_val,
                    Y_val,
                    X_test,
                    Y_test,
                    x,
                    days[0][EMA_INDEX]
                )


    model = Sequential()
    model.add(Dense(1024, input_dim=len(X_train[0]), activation='relu'))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(1,
                activation='softmax',
                kernel_regularizer=L1L2(l1=0.0, l2=0.15)))

    model.compile(optimizer='rmsprop',
              loss='mean_squared_error',
              metrics=['acc'])
    model.fit(X_train, Y_train, epochs=15,  validation_data=(X_val, Y_val))

    y_pred = model.predict(X_test)
    return y_pred, Y_test





def regression_participant_dependent(csv, normalizerMethod, normalizer1, normalizer2, totalDays):
    X_train = []
    Y_train = []

    X_val = []
    Y_val = []

    X_test = []
    Y_test = []


    for i in range(totalDays - 1, len(csv)):
        days = setupTrain.collectDayData(csv, i, totalDays)

        if setupTrain.isSameUserAcross(days):
            x = setupTrain.transform_into_x_feature(
                days,
                True,
                "simpleNN",
                totalDays,
                "z-score",
                normalizer1,
                normalizer2
            )

            if days[0][EMA_INDEX] != '':
                X_train, Y_train, X_val, Y_val, X_test, Y_test = setupTrain.collect_train_val_test_dependent(
                    False,
                    0.60,
                    0.75,
                    X_train,
                    Y_train,
                    X_val,
                    Y_val,
                    X_test,
                    Y_test,
                    x,
                    days[0][EMA_INDEX]
                )


    model = Sequential()
    model.add(Dense(1024, input_dim=len(X_train[0]), activation='relu'))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(1,
                activation='relu',
                kernel_regularizer=L1L2(l1=0.0, l2=0.2)))

    model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['acc'])
    model.fit(X_train, Y_train, epochs=15,  validation_data=(X_val, Y_val))

    y_pred = model.predict(X_test)
    print(y_pred[:10])
    return y_pred, Y_test