import numpy as np
import keras
import random
import math

from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, TimeDistributed, Conv2D, Convolution1D, Conv1D
from keras.regularizers import L1L2

from utilities import model_utilities
from utilities import setupTrain

# index 93 is ema_CALM
EMA_INDEX = 93

def classify_participant_dependent(csv, trainingid, validationid, testingid, maximum, minimum, totalDays):
    X_train = []
    Y_train = []

    X_val = []
    Y_val = []

    X_test = []
    Y_test = []

    for i in range(totalDays, len(csv)):
        days = setupTrain.collectDayData(csv, i, totalDays)

        if setupTrain.isSameUserAcross(days):
            x = setupTrain.transform_into_x_feature(
                days,
                False,
                "LSTM",
                totalDays,
                maximum,
                minimum
            )

            # put the x vector into the appropriate set (i.e. training, validation, testing)
            if days[0][EMA_INDEX] != '':

                X_train, Y_train, X_val, Y_val, X_test, Y_test = setupTrain.collect_train_val_test_dependent(
                    True,
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
    model.add(LSTM(64, return_sequences=True, input_shape=(totalDays, len(X_train[0][0]))))
    model.add(LSTM(64, return_sequences=True, recurrent_dropout=0.2))
    model.add(LSTM(64))

    model.add(Dense(4, activation='softmax', kernel_regularizer=L1L2(l1=0.0, l2=0.0)))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['categorical_accuracy'])
    model.fit(X_train, Y_train, epochs=30,  validation_data=(X_val, Y_val))

    y_pred = model.predict(X_test)
    return model_utilities.convert_preds_into_ema(y_pred), Y_test



def classify_participant_independent(csv, maximum, minimum, totalDays):
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
                False,
                "LSTM",
                totalDays,
                maximum,
                minimum
            )

            if days[0][EMA_INDEX] != '':
                X_train, Y_train, X_val, Y_val, X_test, Y_test = setupTrain.collect_train_val_test_independent(
                    True,
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

    model.add(LSTM(128, return_sequences=True, input_shape=(totalDays, len(X_train[0][0]))))
    model.add(LSTM(128, return_sequences=True, recurrent_dropout=0.2))
    #model.add(LSTM(128, return_sequences=True, recurrent_dropout=0.2))
    model.add(LSTM(64))

    model.add(Dense(4, activation='softmax', kernel_regularizer=L1L2(l1=0.0, l2=0.0)))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['categorical_accuracy'])
    model.fit(X_train, Y_train, epochs=30, validation_data=(X_val, Y_val))

    y_pred = model.predict(X_test)
    # return the predictions and the truth values
    return model_utilities.convert_preds_into_ema(y_pred), Y_test
