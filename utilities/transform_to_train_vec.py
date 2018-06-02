import numpy as np
import math

# converts into a final feature vector for the model to train on
def transform(daysData, isFlatten, modelType, numDays, normalizerMethod, normalizer1, normalizer2):

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


        normalizer1_vec = np.concatenate((normalizer1[33:93],normalizer1[132:]))
        normalizer2_vec = np.concatenate((normalizer2[33:93],normalizer2[132:]))


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

        normalizer1_vec = np.concatenate((normalizer1[33:93],normalizer1[132:]))
        normalizer2_vec = np.concatenate((normalizer2[33:93],normalizer2[132:]))

    elif modelType == "LSTM":

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
        normalizer1_vec = []
        normalizer2_vec = []

        for i in range(0, 90):
            feat_idx = select_features[i]
            if curr is not None:
                features_curr = np.append(features_curr, curr[feat_idx])
            if one_prev is not None:
                features_one_prev = np.append(features_one_prev, one_prev[feat_idx])
            if two_prev is not None:
                features_two_prev = np.append(features_two_prev, two_prev[feat_idx])
            if three_prev is not None:
                features_three_prev = np.append(features_three_prev, three_prev[feat_idx])
            if four_prev is not None:
                features_four_prev = np.append(features_four_prev, four_prev[feat_idx])
            if five_prev is not None:
                features_five_prev = np.append(features_five_prev, five_prev[feat_idx])
            if six_prev is not None:
                features_six_prev = np.append(features_six_prev, six_prev[feat_idx])
            if seven_prev is not None:
                features_seven_prev = np.append(features_seven_prev, seven_prev[feat_idx])

            normalizer1_vec = np.append(normalizer1_vec, normalizer1[feat_idx])
            normalizer2_vec = np.append(normalizer2_vec, normalizer2[feat_idx])

        x_curr = features_curr
        x_one_prev = features_one_prev
        x_two_prev = features_two_prev
        x_three_prev = features_three_prev
        x_four_prev = features_four_prev
        x_five_prev = features_five_prev
        x_six_prev = features_six_prev
        x_seven_prev = features_seven_prev
        '''



        if curr is not None:
            x_curr = np.concatenate((curr[23:93], curr[122:]))
        if one_prev is not None:
            x_one_prev = np.concatenate((one_prev[23:93], one_prev[122:]))
        if two_prev is not None:
            x_two_prev = np.concatenate((two_prev[23:93], two_prev[122:]))
        if three_prev is not None:
            x_three_prev = np.concatenate((three_prev[23:93], three_prev[122:]))
        if four_prev is not None:
            x_four_prev = np.concatenate((four_prev[23:93], four_prev[122:]))
        if five_prev is not None:
            x_five_prev = np.concatenate((five_prev[23:93], five_prev[122:]))
        if six_prev is not None:
            x_six_prev = np.concatenate((six_prev[23:93], six_prev[122:]))
        if seven_prev is not None:
            x_seven_prev = np.concatenate((seven_prev[23:93], seven_prev[122:]))

        normalizer1_vec = np.concatenate((normalizer1[23:93],normalizer1[122:]))
        normalizer2_vec = np.concatenate((normalizer2[23:93],normalizer2[122:]))




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



    # if normalizerMethod is z-score
    # normalizer1 = mean_features
    # normalizer2 = std_features

    #if normalizerMethod = max-min
    # normalizer1 = minimum_features
    # normalizer2 = maximum_features
    if numDays >= 1:
        if normalizerMethod == "z-score":
            ret1 = np.divide(np.subtract(numeric_x_curr, normalizer1_vec), normalizer2_vec)
        elif normalizerMethod == "max-min":
            ret1 = np.divide(np.subtract(numeric_x_curr, normalizer1_vec), np.subtract(normalizer2_vec, normalizer1_vec))
        x[0][0] = np.array([0.0 if math.isnan(float(num)) else num for num in ret1])

    if numDays >= 2:
        if normalizerMethod == "z-score":
            ret2 = np.divide(np.subtract(numeric_x_one_prev, normalizer1_vec), normalizer2_vec)
        elif normalizerMethod == "max-min":
            ret2 = np.divide(np.subtract(numeric_x_one_prev, normalizer1_vec), np.subtract(normalizer2_vec, normalizer1_vec))
        x[0][1] = np.array([0.0 if math.isnan(float(num)) else num for num in ret2])

    if numDays >= 3:
        if normalizerMethod == "z-score":
            ret3 = np.divide(np.subtract(numeric_x_two_prev, normalizer1_vec), normalizer2_vec)
        elif normalizerMethod == "max-min":
            ret3 = np.divide(np.subtract(numeric_x_two_prev, normalizer1_vec), np.subtract(normalizer2_vec, normalizer1_vec))
        x[0][2] = np.array([0.0 if math.isnan(float(num)) else num for num in ret3])

    if numDays >= 4:
        if normalizerMethod == "z-score":
            ret4 = np.divide(np.subtract(numeric_x_three_prev, normalizer1_vec), normalizer2_vec)
        elif normalizerMethod == "max-min":
            ret4 = np.divide(np.subtract(numeric_x_three_prev, normalizer1_vec), np.subtract(normalizer2_vec, normalizer1_vec))
        x[0][3] = np.array([0.0 if math.isnan(float(num)) else num for num in ret4])

    if numDays >= 5:
        if normalizerMethod == "z-score":
            ret5 = np.divide(np.subtract(numeric_x_four_prev, normalizer1_vec), normalizer2_vec)
        elif normalizerMethod == "max-min":
            ret5 = np.divide(np.subtract(numeric_x_four_prev, normalizer1_vec), np.subtract(normalizer2_vec, normalizer1_vec))
        x[0][4] = np.array([0.0 if math.isnan(float(num)) else num for num in ret5])

    if numDays >= 6:
        if normalizerMethod == "z-score":
            ret6 = np.divide(np.subtract(numeric_x_five_prev, normalizer1_vec), normalizer2_vec)
        elif normalizerMethod == "max-min":
            ret6 = np.divide(np.subtract(numeric_x_five_prev, normalizer1_vec), np.subtract(normalizer2_vec, normalizer1_vec))
        x[0][5] = np.array([0.0 if math.isnan(float(num)) else num for num in ret6])

    if numDays >= 7:
        if normalizerMethod == "z-score":
            ret7 = np.divide(np.subtract(numeric_x_six_prev, normalizer1_vec), normalizer2_vec)
        elif normalizerMethod == "max-min":
            ret7 = np.divide(np.subtract(numeric_x_six_prev, normalizer1_vec), np.subtract(normalizer2_vec, normalizer1_vec))
        x[0][6] = np.array([0.0 if math.isnan(float(num)) else num for num in ret7])

    if numDays >= 8:
        if normalizerMethod == "z-score":
            ret8 = np.divide(np.subtract(numeric_x_seven_prev, normalizer1_vec), normalizer2_vec)
        elif normalizerMethod == "max-min":
            ret8 = np.divide(np.subtract(numeric_x_seven_prev, normalizer1_vec), np.subtract(normalizer2_vec, normalizer1_vec))
        x[0][7] = np.array([0.0 if math.isnan(float(num)) else num for num in ret8])

    if isFlatten:
        return [x.flatten()]
    else:
        return x






