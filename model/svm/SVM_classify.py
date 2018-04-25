import numpy as np
from sklearn import svm
from sklearn.svm import LinearSVC
import random

from utilities import model_utilities
from utilities import setupTrain

EMA_INDEX = 93

def classification_participant_dependent(csv, trainingid, validationid, testingid, maximum, minimum, totalDays):
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
                "SVM",
                totalDays,
                maximum,
                minimum
            )

            # put the x vector into the appropriate set (i.e. training, validation, testing)
            if days[0][EMA_INDEX] != '':
                X_train, Y_train, X_val, Y_val, X_test, Y_test = setupTrain.collect_train_val_test_dependent(
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

    clf = LinearSVC(random_state=0)
    clf.fit(X_train, Y_train)
    preds = clf.predict(X_test)
    return preds, Y_test


def classification_participant_independent(csv, maximum, minimum, totalDays):
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
                "SVM",
                totalDays,
                maximum,
                minimum
            )

            if days[0][EMA_INDEX] != '':
                X_train, Y_train, X_val, Y_val, X_test, Y_test = setupTrain.collect_train_val_test_independent(
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

    clf = LinearSVC(random_state=0, C=2)
    #clf = svm.SVC(random_state=0, C= 10, kernel = 'linear')
    clf.fit(X_train, Y_train)
    preds = clf.predict(X_test)
    return preds, Y_test
