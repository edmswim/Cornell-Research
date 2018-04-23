import numpy as np
import keras
import random
import math

from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, TimeDistributed, Conv2D, Convolution1D, Conv1D
from keras.regularizers import L1L2

import model_utilities

# index 93 is ema_CALM
EMA_INDEX = 93

def LSTM_classification_participant_dependent(csv, trainingid, validationid, testingid, maximum, minimum, totalDays):
    X_train = []
    Y_train = []

    X_val = []
    Y_val = []

    X_test = []
    Y_test = []

    for i in range(totalDays, len(csv)):
        days = []

        if totalDays >= 1:
            days = np.array([csv[i]])

        if totalDays >= 2:
            days = np.append(days, [csv[i-1]], axis=0)

        if totalDays >= 3:
            days = np.append(days, [csv[i-2]], axis=0)

        if totalDays >= 4:
            days = np.append(days, [csv[i-3]], axis=0)

        if totalDays >= 5:
            days = np.append(days, [csv[i-4]], axis=0)

        if totalDays >= 6:
            days = np.append(days, [csv[i-5]], axis=0)

        if totalDays >= 7:
            days = np.append(days, [csv[i-6]], axis=0)

        if totalDays >= 8:
            days = np.append(days, [csv[i-7]], axis=0)

        if model_utilities.isSameUserAcross(days):
            x = model_utilities.transform_into_x_feature(
                days,
                False,
                "LSTM",
                totalDays,
                maximum,
                minimum
            )

            # put the x vector into the appropriate set (i.e. training, validation, testing)
            if days[0][EMA_INDEX] != '':
                #training
                if days[0][1] in trainingid:
                    if len(X_train) == 0:
                        X_train = x
                    else:
                        X_train = np.concatenate((X_train, x), axis=0)

                    one_hot_labels = [keras.utils.to_categorical(
                        int(days[0][EMA_INDEX]), num_classes=4)]
                    if len(Y_train) == 0:
                        Y_train = one_hot_labels
                    else:
                        Y_train = np.concatenate((Y_train, one_hot_labels), axis=0)

                # validation
                elif days[0][1] in validationid:
                    if len(X_val) == 0:
                        X_val = x
                    else:
                        X_val = np.concatenate((X_val, x), axis=0)

                    one_hot_labels = [keras.utils.to_categorical(
                        int(days[0][EMA_INDEX]), num_classes=4)]
                    if len(Y_val) == 0:
                        Y_val = one_hot_labels
                    else:
                        Y_val = np.concatenate((Y_val, one_hot_labels), axis=0)

                # testing
                elif days[0][1] in testingid:
                    if len(X_test) == 0:
                        X_test = x
                    else:
                        X_test = np.concatenate((X_test, x), axis=0)

                    Y_test = np.append(Y_test, int(days[0][EMA_INDEX]))


    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=(totalDays, len(X_train[0][0]))))
    model.add(LSTM(64, return_sequences=True, recurrent_dropout=0.2))
    model.add(LSTM(64))

    model.add(Dense(4, activation='softmax', kernel_regularizer=L1L2(l1=0.0, l2=0.0)))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['categorical_accuracy'])
    model.fit(X_train, Y_train, epochs=30,  validation_data=(X_val, Y_val))

    y_pred = model.predict(X_test)
    return model_utilities.convert_preds_into_ema(y_pred), Y_test






def LSTM_classification_participant_independent(csv, maximum, minimum, totalDays):
    X_train = []
    Y_train = []

    X_val = []
    Y_val = []

    X_test = []
    Y_test = []

    for i in range(totalDays - 1, len(csv)):
        days = []
        if totalDays >= 1:
            days = np.array([csv[i]])

        if totalDays >= 2:
            days = np.append(days, [csv[i-1]], axis=0)

        if totalDays >= 3:
            days = np.append(days, [csv[i-2]], axis=0)

        if totalDays >= 4:
            days = np.append(days, [csv[i-3]], axis=0)

        if totalDays >= 5:
            days = np.append(days, [csv[i-4]], axis=0)

        if totalDays >= 6:
            days = np.append(days, [csv[i-5]], axis=0)

        if totalDays >= 7:
            days = np.append(days, [csv[i-6]], axis=0)

        if totalDays >= 8:
            days = np.append(days, [csv[i-7]], axis=0)


        if model_utilities.isSameUserAcross(days):
            x = model_utilities.transform_into_x_feature(
                days,
                False,
                "LSTM",
                totalDays,
                maximum,
                minimum
            )

            if days[0][EMA_INDEX] != '':
                p = random.random()
                # put the x vector into the appropriate set (i.e. training, validation, testing)
                if p <= 0.60:
                    #training
                    if len(X_train) == 0:
                        X_train = x
                    else:
                        X_train = np.concatenate((X_train, x), axis=0)

                    one_hot_labels = [keras.utils.to_categorical(
                        int(days[0][EMA_INDEX]), num_classes=4)]
                    if len(Y_train) == 0:
                        Y_train = one_hot_labels
                    else:
                        Y_train = np.concatenate((Y_train, one_hot_labels), axis=0)

                elif p > 0.60 and p <= 0.75:
                    # validation
                    if len(X_val) == 0:
                        X_val = x
                    else:
                        X_val = np.concatenate((X_val, x), axis=0)

                    one_hot_labels = [keras.utils.to_categorical(
                        int(days[0][EMA_INDEX]), num_classes=4)]
                    if len(Y_val) == 0:
                        Y_val = one_hot_labels
                    else:
                        Y_val = np.concatenate((Y_val, one_hot_labels), axis=0)
                else:
                    # testing
                    if len(X_test) == 0:
                        X_test = x
                    else:
                        X_test = np.concatenate((X_test, x), axis=0)

                    #the truth values
                    Y_test = np.append(Y_test, int(days[0][EMA_INDEX]))


    model = Sequential()

    model.add(LSTM(128, return_sequences=True, input_shape=(totalDays, len(X_train[0][0]))))
    model.add(LSTM(128, return_sequences=True, recurrent_dropout=0.2))
    model.add(LSTM(128, return_sequences=True, recurrent_dropout=0.2))
    model.add(LSTM(64))

    model.add(Dense(4, activation='softmax', kernel_regularizer=L1L2(l1=0.0, l2=0.0)))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['categorical_accuracy'])
    model.fit(X_train, Y_train, epochs=30, validation_data=(X_val, Y_val))

    y_pred = model.predict(X_test)
    # return the predictions and the truth values
    return model_utilities.convert_preds_into_ema(y_pred), Y_test
