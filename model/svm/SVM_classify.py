import numpy as np
from sklearn import svm
from sklearn.svm import LinearSVC
import random

import model_utilities

EMA_INDEX = 93

def classification_participant_dependent(csv, trainingid, validationid, testingid, maximum, minimum, totalDays):
    X_train = []
    Y_train = []

    X_val = []
    Y_val = []

    X_test = []
    Y_test = []

    for i in range(totalDays - 1, len(csv)):
        days = model_utilities.collectDayData(csv, i, totalDays)

        if model_utilities.isSameUserAcross(days):
            x = model_utilities.transform_into_x_feature(
                days,
                True,
                "SVM",
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

                    Y_train = np.append(Y_train, int(days[0][EMA_INDEX]))


                # validation
                if days[0][1] in validationid:
                    if len(X_val) == 0:
                        X_val = x
                    else:
                        X_val = np.concatenate((X_val, x), axis=0)

                    Y_val = np.append(Y_val, int(days[0][EMA_INDEX]))

                # testing
                if days[0][1] in testingid:
                    if len(X_test) == 0:
                        X_test = x
                    else:
                        X_test = np.concatenate((X_test, x), axis=0)

                    Y_test = np.append(Y_test, int(days[0][EMA_INDEX]))

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
        days = model_utilities.collectDayData(csv, i, totalDays)

        if model_utilities.isSameUserAcross(days):
            x = model_utilities.transform_into_x_feature(
                days,
                True,
                "SVM",
                totalDays,
                maximum,
                minimum
            )

            if days[0][EMA_INDEX] != '':
                p = np.random.uniform(0.0, 1.0, 1)
                # put the x vector into the appropriate set (i.e. training, validation, testing)
                if p <= 0.6:
                    #training
                    if len(X_train) == 0:
                        X_train = x
                    else:
                        X_train = np.concatenate((X_train, x), axis=0)

                    Y_train = np.append(Y_train, int(days[0][EMA_INDEX]))

                elif p > 0.6 and p <= 0.75:
                    # validation
                    if len(X_val) == 0:
                        X_val = x
                    else:
                        X_val = np.concatenate((X_val, x), axis=0)

                    Y_val = np.append(Y_val, int(days[0][EMA_INDEX]))
                else:
                    # testing
                    if len(X_test) == 0:
                        X_test = x
                    else:
                        X_test = np.concatenate((X_test, x), axis=0)

                    Y_test = np.append(Y_test, int(days[0][EMA_INDEX]))

    clf = LinearSVC(random_state=0, C=2)
    #clf = svm.SVC(random_state=0, C= 10, kernel = 'linear')
    clf.fit(X_train, Y_train)
    preds = clf.predict(X_test)
    return preds, Y_test
