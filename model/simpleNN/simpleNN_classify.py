import numpy as np
import keras

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, MaxPooling1D, Convolution1D
from keras.regularizers import L1L2

from utilities import prediction_utilities
from utilities import data_collector
from utilities import assign_data
from utilities import transform_to_train_vec


# index 93 is ema_CALM
EMA_INDEX = 93

def classification_participant_independent(csv, trainingid, validationid, testingid, normalizerMethod, normalizer1, normalizer2, totalDays):
    X_train = []
    Y_train = []

    X_val = []
    Y_val = []

    X_test = []
    Y_test = []

    for i in range(totalDays - 1, len(csv)):
        days = data_collector.collectDayData(csv, i, totalDays)

        if data_collector.isSameUserAcross(days):
            x = transform_to_train_vec.transform(
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
                X_train, Y_train, X_val, Y_val, X_test, Y_test = assign_data.independent_assign(
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
    model.add(Dense(1024, input_dim=len(X_train[0]), activation='relu'))
    model.add(Dense(256, activation='relu'))

    model.add(Dense(4,
                activation='softmax',
                kernel_regularizer=L1L2(l1=0.0, l2=0.15)))

    model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    model.fit(X_train, Y_train, epochs=30,  validation_data=(X_val, Y_val))

    y_pred = model.predict(X_test)
    return prediction_utilities.convert_preds_into_ema(y_pred), Y_test




def classification_participant_dependent(csv, normalizerMethod, normalizer1, normalizer2, totalDays, leave_one_patient):
    X_train = []
    Y_train = []

    X_val = []
    Y_val = []

    X_test = []
    Y_test = []


    for i in range(totalDays - 1, len(csv)):
        days = data_collector.collectDayData(csv, i, totalDays)
        userid = days[0][1]

        if data_collector.isSameUserAcross(days):
            x = transform_to_train_vec.transform(
                days,
                True,
                "simpleNN",
                totalDays,
                "z-score",
                normalizer1,
                normalizer2
            )

            if days[0][EMA_INDEX] != '':
                X_train, Y_train, X_val, Y_val, X_test, Y_test = assign_data.dependent_assign(
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
                    days[0][EMA_INDEX],
                    leave_one_patient,
                    userid
                )

    model = Sequential()
    model.add(Dense(1024, input_dim=len(X_train[0]), activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(32, activation='relu'))
    #model.add(Dropout(0.4))

    model.add(Dense(4,
                activation='softmax',
                kernel_regularizer=L1L2(l1=0.0, l2=0.25)))

    model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['categorical_accuracy'])


    if len(X_val) == 0:
        model.fit(X_train, Y_train, epochs=4)
    else:
        model.fit(X_train, Y_train, epochs=4,  validation_data=(X_val, Y_val))

    y_pred = model.predict(X_test)
    return prediction_utilities.convert_preds_into_ema(y_pred), Y_test



