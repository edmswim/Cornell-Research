import numpy as np
import math
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_selection import chi2

import keras

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression




#convert the prediction prob vec into ema scores
def convert_preds_into_ema(preds):
    pred_final = []
    # transform the predicted probability vector into an ema score by taking the index with the highest probability
    for i in range(len(preds)):
        ema_score = np.argmax(preds[i])
        pred_final = np.append(pred_final, ema_score)
    return pred_final





# finds maximum and minimum of features across the entire dataset
def find_max_min(csv):
    ema_scores = [93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106]
    csv = np.array(csv)
    # first three columns are studyid, eurekaid, timestamp (which are useless) so just convert to 0 since we aren't normalizing this
    maximum = np.array([0, 0, 0])
    minimum = np.array([0, 0, 0])
    for i in range(3, csv.shape[1]):
        # if it's ema scores, just convert to 0 because we don't care since we aren't normalizing this
        if i in ema_scores:
            maximum = np.append(maximum, 0)
            minimum = np.append(minimum, 0)
        else:
            arr = np.array([row[i] for row in csv])
            num_arr = np.array([0.0 if math.isnan(float(numeric_string)) else float(numeric_string) for numeric_string in arr])
            maximum = np.append(maximum, np.amax(num_arr, axis=0))
            minimum = np.append(minimum, np.amin(num_arr, axis=0))
    return maximum, minimum





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







