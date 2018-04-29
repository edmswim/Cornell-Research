import numpy as np
from sklearn import svm
from sklearn.svm import LinearSVC

from utilities import model_utilities
from utilities import setupTrain

EMA_INDEX = 93

def classification_participant_independent(csv, trainingid, validationid, testingid, normalizerMethod, normalizer1, normalizer2, totalDays):
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

    clf = LinearSVC(random_state=0, class_weight={0: 0.3, 1: 0.25, 2: 0.25, 3: 0.20})
    clf.fit(X_train, Y_train)
    preds = clf.predict(X_test)
    return preds, Y_test


def classification_participant_dependent(csv, normalizerMethod, normalizer1, normalizer2, totalDays):
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

    # FIX class_weight
    clf = LinearSVC(random_state=0, C = 2)
    #clf = svm.SVC(random_state=0, class_weight={0:0.3, 1:0.25, 2:0.2, 3:0.2})

    clf.fit(X_train, Y_train)
    preds = clf.predict(X_test)
    return preds, Y_test
