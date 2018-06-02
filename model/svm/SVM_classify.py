import numpy as np
from sklearn import svm
from sklearn.svm import LinearSVC
from utilities import prediction_utilities
from utilities import assign_data
from utilities import transform_to_train_vec

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
                "SVM",
                totalDays,
                "z-score",
                normalizer1,
                normalizer2
            )

            # put the x vector into the appropriate set (i.e. training, validation, testing)
            if days[0][EMA_INDEX] != '':
                X_train, Y_train, X_val, Y_val, X_test, Y_test = assign_data.independent_assign(
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
                "SVM",
                totalDays,
                "z-score",
                normalizer1,
                normalizer2
            )

            if days[0][EMA_INDEX] != '':
                X_train, Y_train, X_val, Y_val, X_test, Y_test = assign_data.dependent_assign(
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
                    days[0][EMA_INDEX],
                    leave_one_patient,
                    userid
                )

    # FIX class_weight
    clf = LinearSVC(random_state=0, C=1)
    #clf = svm.SVC(random_state=0, class_weight={0:0.3, 1:0.25, 2:0.2, 3:0.2})

    clf.fit(X_train, Y_train)
    preds = clf.predict(X_test)
    return preds, Y_test
