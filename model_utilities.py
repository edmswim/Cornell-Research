import numpy as np
import math
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_selection import chi2

import keras

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

# collects features of current day/previous days
def collectDayData(csv, index, totalDays):
    days = []

    if totalDays >= 1:
        days = np.array([csv[index]])

    if totalDays >= 2:
        days = np.append(days, [csv[index-1]], axis=0)

    if totalDays >= 3:
        days = np.append(days, [csv[index-2]], axis=0)

    if totalDays >= 4:
        days = np.append(days, [csv[index-3]], axis=0)

    if totalDays >= 5:
        days = np.append(days, [csv[index-4]], axis=0)

    if totalDays >= 6:
        days = np.append(days, [csv[index-5]], axis=0)

    if totalDays >= 7:
        days = np.append(days, [csv[index-6]], axis=0)

    if totalDays >= 8:
        days = np.append(days, [csv[index-7]], axis=0)

    return days


# converts into a final feature vector for the model to train on
def transform_into_x_feature(daysData, isFlatten, modelType, numDays, maximum, minimum):

    if daysData.shape[0] >= 1:
        curr = daysData[0]
    else:
        curr = None

    if daysData.shape[0] >= 2:
        one_prev = daysData[1]
    else:
        one_prev = None

    if daysData.shape[0] >= 3:
        two_prev = daysData[2]
    else:
        two_prev = None

    if daysData.shape[0] >= 4:
        three_prev = daysData[3]
    else:
        three_prev = None

    if daysData.shape[0] >= 5:
        four_prev = daysData[4]
    else:
        four_prev = None

    if daysData.shape[0] >= 6:
        five_prev = daysData[5]
    else:
        five_prev = None

    if daysData.shape[0] >= 7:
        six_prev = daysData[6]
    else:
        six_prev = None

    if daysData.shape[0] >= 8:
        seven_prev = daysData[7]
    else:
        seven_prev = None


    if modelType == "simpleNN":
        if curr is not None:
            x_curr = np.concatenate((curr[33:93], curr[132:]))
        if one_prev is not None:
            x_one_prev = np.concatenate((one_prev[33:93], one_prev[132:]))
        if two_prev is not None:
            x_two_prev = np.concatenate((two_prev[33:93], two_prev[132:]))
        if three_prev is not None:
            x_three_prev = np.concatenate((three_prev[33:93], three_prev[132:]))
        if four_prev is not None:
            x_four_prev = np.concatenate((four_prev[33:93], four_prev[132:]))
        if five_prev is not None:
            x_five_prev = np.concatenate((five_prev[33:93], five_prev[132:]))
        if six_prev is not None:
            x_six_prev = np.concatenate((six_prev[33:93], six_prev[132:]))
        if seven_prev is not None:
            x_seven_prev = np.concatenate((seven_prev[33:93], seven_prev[132:]))

        minimum_vec = np.concatenate((minimum[33:93],minimum[132:]))
        maximum_vec = np.concatenate((maximum[33:93],maximum[132:]))

    elif modelType == "SVM":
        if curr is not None:
            x_curr = np.concatenate((curr[33:93], curr[132:]))
        if one_prev is not None:
            x_one_prev = np.concatenate((one_prev[33:93], one_prev[132:]))
        if two_prev is not None:
            x_two_prev = np.concatenate((two_prev[33:93], two_prev[132:]))
        if three_prev is not None:
            x_three_prev = np.concatenate((three_prev[33:93], three_prev[132:]))
        if four_prev is not None:
            x_four_prev = np.concatenate((four_prev[33:93], four_prev[132:]))
        if five_prev is not None:
            x_five_prev = np.concatenate((five_prev[33:93], five_prev[132:]))
        if six_prev is not None:
            x_six_prev = np.concatenate((six_prev[33:93], six_prev[132:]))
        if seven_prev is not None:
            x_seven_prev = np.concatenate((seven_prev[33:93], seven_prev[132:]))

        minimum_vec = np.concatenate((minimum[33:93],minimum[132:]))
        maximum_vec = np.concatenate((maximum[33:93],maximum[132:]))

    elif modelType == "LSTM":


        # TRYING TO DO FEATURE SELECTION
        '''
        select_features = np.array([
            58,
            35,
            70,
            53,
            59,
            15,
            60,
            13,
            68,
            47,
            145,
            150,
            131,
            43,
            61,
            73,
            23,
            149,
            89,
            57,
            33,
            148,
            154,
            152,
            56,
            16,
            142,
            90,
            51,
            44,
            25,
            48,
            62,
            46,
            9,
            143,
            77,
            153,
            50,
            54,
            71,
            140,
            84,
            45,
            136,
            87,
            26,
            75,
            141,
            138,
            14,
            135,
            49,
            63,
            52,
            132,
            27,
            88,
            67,
            147,
            3,
            137,
            72,
            129,
            130,
            65,
            24,
            83,
            64,
            55,
            36,
            124,
            76,
            146,
            151,
            7,
            85,
            118,
            5,
            4,
            34,
            125,
            74,
            121,
            66,
            17,
            109,
            122,
            37,
            114,
            139,
            69,
            6,
            133,
            144,
            107,
            128,
            86,
            127,
            92,
            123,
            126,
            112,
            134,
            119,
            8,
            113,
            10,
            111,
            11,
            29,
            110,
            117,
            91,
            30,
            115,
            116,
            12,
            31,
            32,
            108,
            28,
            120
        ])

        features_curr = []
        features_one_prev = []
        features_two_prev = []
        features_three_prev = []
        features_four_prev = []
        features_five_prev = []
        features_six_prev = []
        features_seven_prev = []

        minimum_vec = []
        maximum_vec = []

        for i in range(0, 100):
            feat_idx = select_features[i]

            if curr is not None:
                features_curr = np.append(features_curr, curr[feat_idx])
            if one_prev is not None:
                features_one_prev = np.append(features_one_prev, curr[feat_idx])
            if two_prev is not None:
                features_two_prev = np.append(features_two_prev, curr[feat_idx])
            if three_prev is not None:
                features_three_prev = np.append(features_three_prev, curr[feat_idx])
            if four_prev is not None:
                features_four_prev = np.append(features_four_prev, curr[feat_idx])
            if five_prev is not None:
                features_five_prev = np.append(features_five_prev, curr[feat_idx])
            if six_prev is not None:
                features_six_prev = np.append(features_six_prev, curr[feat_idx])
            if seven_prev is not None:
                features_seven_prev = np.append(features_seven_prev, curr[feat_idx])

            minimum_vec = np.append(minimum_vec, minimum[feat_idx])
            maximum_vec = np.append(maximum_vec, maximum[feat_idx])

        x_curr = features_curr
        x_one_prev = features_one_prev
        x_two_prev = features_two_prev
        x_three_prev = features_three_prev
        x_four_prev = features_four_prev
        x_five_prev = features_five_prev
        x_six_prev = features_six_prev
        x_seven_prev = features_seven_prev
        '''
        # END FEATURE SELECTION


        if curr is not None:
            x_curr = np.concatenate((curr[33:93], curr[132:]))
        if one_prev is not None:
            x_one_prev = np.concatenate((one_prev[33:93], one_prev[132:]))
        if two_prev is not None:
            x_two_prev = np.concatenate((two_prev[33:93], two_prev[132:]))
        if three_prev is not None:
            x_three_prev = np.concatenate((three_prev[33:93], three_prev[132:]))
        if four_prev is not None:
            x_four_prev = np.concatenate((four_prev[33:93], four_prev[132:]))
        if five_prev is not None:
            x_five_prev = np.concatenate((five_prev[33:93], five_prev[132:]))
        if six_prev is not None:
            x_six_prev = np.concatenate((six_prev[33:93], six_prev[132:]))
        if seven_prev is not None:
            x_seven_prev = np.concatenate((seven_prev[33:93], seven_prev[132:]))

        minimum_vec = np.concatenate((minimum[33:93],minimum[132:]))
        maximum_vec = np.concatenate((maximum[33:93],maximum[132:]))



    # convert them from string to float and taking care of NaNs
    if curr is not None:
        numeric_x_curr = np.array([0.0 if math.isnan(float(numeric_string)) else float(numeric_string) for numeric_string in x_curr])
    if one_prev is not None:
        numeric_x_one_prev = np.array([0.0 if math.isnan(float(numeric_string)) else float(numeric_string) for numeric_string in x_one_prev])
    if two_prev is not None:
        numeric_x_two_prev = np.array([0.0 if math.isnan(float(numeric_string)) else float(numeric_string) for numeric_string in x_two_prev])
    if three_prev is not None:
        numeric_x_three_prev = np.array([0.0 if math.isnan(float(numeric_string)) else float(numeric_string) for numeric_string in x_three_prev])
    if four_prev is not None:
        numeric_x_four_prev = np.array([0.0 if math.isnan(float(numeric_string)) else float(numeric_string) for numeric_string in x_four_prev])
    if five_prev is not None:
        numeric_x_five_prev = np.array([0.0 if math.isnan(float(numeric_string)) else float(numeric_string) for numeric_string in x_five_prev])
    if six_prev is not None:
        numeric_x_six_prev = np.array([0.0 if math.isnan(float(numeric_string)) else float(numeric_string) for numeric_string in x_six_prev])
    if seven_prev is not None:
        numeric_x_seven_prev = np.array([0.0 if math.isnan(float(numeric_string)) else float(numeric_string) for numeric_string in x_seven_prev])


    # final feature vector (can vary how many past days to use)
    x = np.zeros((1, numDays, len(x_curr)))

    if numDays >= 1:
        ret1 = np.divide(np.subtract(numeric_x_curr, minimum_vec), np.subtract(maximum_vec, minimum_vec))
        x[0][0] = np.array([0.0 if math.isnan(float(num)) else num for num in ret1])

    if numDays >= 2:
        ret2 = np.divide(np.subtract(numeric_x_one_prev, minimum_vec), np.subtract(maximum_vec, minimum_vec))
        x[0][1] = np.array([0.0 if math.isnan(float(num)) else num for num in ret2])

    if numDays >= 3:
        ret3 = np.divide(np.subtract(numeric_x_two_prev, minimum_vec), np.subtract(maximum_vec, minimum_vec))
        x[0][2] = np.array([0.0 if math.isnan(float(num)) else num for num in ret3])

    if numDays >= 4:
        ret4 = np.divide(np.subtract(numeric_x_three_prev, minimum_vec), np.subtract(maximum_vec, minimum_vec))
        x[0][3] = np.array([0.0 if math.isnan(float(num)) else num for num in ret4])

    if numDays >= 5:
        ret5 = np.divide(np.subtract(numeric_x_four_prev, minimum_vec), np.subtract(maximum_vec, minimum_vec))
        x[0][4] = np.array([0.0 if math.isnan(float(num)) else num for num in ret5])

    if numDays >= 6:
        ret6 = np.divide(np.subtract(numeric_x_five_prev, minimum_vec), np.subtract(maximum_vec, minimum_vec))
        x[0][5] = np.array([0.0 if math.isnan(float(num)) else num for num in ret6])

    if numDays >= 7:
        ret7 = np.divide(np.subtract(numeric_x_six_prev, minimum_vec), np.subtract(maximum_vec, minimum_vec))
        x[0][6] = np.array([0.0 if math.isnan(float(num)) else num for num in ret7])

    if numDays >= 8:
        ret8 = np.divide(np.subtract(numeric_x_seven_prev, minimum_vec), np.subtract(maximum_vec, minimum_vec))
        x[0][7] = np.array([0.0 if math.isnan(float(num)) else num for num in ret8])

    if isFlatten:
        return [x.flatten()]
    else:
        return x


#check if the same user is across all the days data
def isSameUserAcross(daysData):
    userid = daysData[0][1]
    for i in range(1, len(daysData)):
        if daysData[i][1] != userid:
            return False
    return True



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







