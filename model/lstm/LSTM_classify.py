import numpy as np
import keras
from utilities import prediction_utilities
from utilities import data_collector
from utilities import assign_data
from utilities import transform_to_train_vec

from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, TimeDistributed, Conv2D, Convolution1D, Conv1D, GRU
from keras.layers import Bidirectional
from keras.regularizers import L1L2


# index 93 is ema_CALM
EMA_INDEX = 93

def classify_participant_independent(csv, trainingid, validationid, testingid, normalizer_type, normalizer1, normalizer2, totalDays):
    X_train = []
    Y_train = []

    X_val = []
    Y_val = []

    X_test = []
    Y_test = []

    for i in range(totalDays, len(csv)):
        days = data_collector.collectDayData(csv, i, totalDays)

        userid = days[0][1]

        if data_collector.isSameUserAcross(days):
            x = transform_to_train_vec.transform(
                days,
                False,
                "LSTM",
                totalDays,
                normalizer_type,
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
    model.add(LSTM(64, return_sequences=True, input_shape=(totalDays, len(X_train[0][0]))))
    model.add(LSTM(64, return_sequences=True, recurrent_dropout=0.2))
    model.add(LSTM(64))

    model.add(Dense(4, activation='softmax', kernel_regularizer=L1L2(l1=0.0, l2=0.0)))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['categorical_accuracy'])

    if len(X_val) == 0:
        model.fit(X_train, Y_train, epochs=5)
    else:
        model.fit(X_train, Y_train, epochs=5,  validation_data=(X_val, Y_val))

    y_pred = model.predict(X_test)
    return prediction_utilities.convert_preds_into_ema(y_pred), Y_test



def classify_participant_dependent(csv, normalizer_type, normalizer1, normalizer2, totalDays, leave_one_patient):
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
                False,
                "LSTM",
                totalDays,
                normalizer_type,
                normalizer1,
                normalizer2
            )





            # x_3_days = setupTrain.transform_into_x_feature(
            #     days,
            #     False,
            #     "LSTM",
            #     4,
            #     "z-score",
            #     normalizer1,
            #     normalizer2
            # )

            # x_1_days = setupTrain.transform_into_x_feature(
            #     days,
            #     False,
            #     "LSTM",
            #     2,
            #     "z-score",
            #     normalizer1,
            #     normalizer2
            # )

            # x_3_days = np.pad(x_3_days, ((0,0), (2,2), (0,0)), 'constant', constant_values = -1)
            # x_1_days = np.pad(x_1_days, ((0,0), (3,3), (0,0)), 'constant', constant_values = -1)


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





                # X_train, Y_train, X_val, Y_val, X_test, Y_test = setupTrain.collect_train_val_test_dependent(
                #     True,
                #     1,
                #     1,
                #     X_train,
                #     Y_train,
                #     X_val,
                #     Y_val,
                #     X_test,
                #     Y_test,
                #     x_3_days,
                #     days[0][EMA_INDEX],
                #     leave_one_patient,
                #     userid
                # )

                # X_train, Y_train, X_val, Y_val, X_test, Y_test = setupTrain.collect_train_val_test_dependent(
                #     True,
                #     1,
                #     1,
                #     X_train,
                #     Y_train,
                #     X_val,
                #     Y_val,
                #     X_test,
                #     Y_test,
                #     x_1_days,
                #     days[0][EMA_INDEX],
                #     leave_one_patient,
                #     userid
                # )

    print(X_test.shape)


    model = Sequential()
    model.add(Bidirectional(LSTM(150, return_sequences=False, recurrent_dropout=0.1), input_shape=(len(X_train[0]), len(X_train[0][0]))))
    model.add(Dense(4, activation='softmax', kernel_regularizer=L1L2(l1=0.0, l2=0.0)))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['categorical_accuracy'])


    if leave_one_patient is None:
        model.fit(X_train, Y_train, epochs=9, validation_data=(X_val, Y_val))
    else:
        model.fit(X_train, Y_train, epochs=9)

    y_pred = model.predict(X_test)
    # return the predictions and the truth values
    return prediction_utilities.convert_preds_into_ema(y_pred), Y_test
