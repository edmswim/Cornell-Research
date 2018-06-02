import numpy as np
import math
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_selection import chi2

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression


def find_important_features(csv, ema_index=93):
    print("Calculating which features are important")
    X = []
    Y = []

    for i in range(0, len(csv)):
        curr = csv[i]
        x_curr = np.array(curr[3:93] + curr[107:])

        numeric_x_curr = np.array([0.0 if math.isnan(float(numeric_string)) else float(numeric_string) for numeric_string in x_curr])


        if curr[ema_index] != '':
            Y = np.append(Y, int(curr[ema_index]))

            if len(X) == 0:
                X = np.array([numeric_x_curr])
                #print(X.shape)
            else:
                X = np.concatenate((X, [numeric_x_curr]), axis=0)
                #print(X.shape)

    print(X.shape)
    test = SelectKBest(f_classif)
    fit = test.fit(X, Y)
    np.set_printoptions(precision=5)
    print(fit.scores_)

    # model = ExtraTreesClassifier()
    # model.fit(X, Y)
    # print(model.feature_importances_)

    # model = LogisticRegression()
    # rfe = RFE(model, 10)
    # fit = rfe.fit(X, Y)

    # print("Selected Features: %s") % fit.support_
    # print("Feature Ranking: %s") % fit.ranking_







