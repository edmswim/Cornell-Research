import numpy as np
import csv
import math
import keras
import argparse
import random

from model.simpleNN import simpleNN_classify
from model.lstm import LSTM_classify
from model.svm import SVM_classify

from model.lstm import LSTM_regression

import metrics
import model_utilities





# reads the csv and returns it as an array
def read_csv(filename):
    with open(filename, 'rb') as csvfile:
        try:
            reader = csv.reader(csvfile, delimiter=',')
            next(reader, None)  # skip the header
            rows = [r for r in reader]
            return rows
        finally:
            csvfile.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='simpleNN')
    parser.add_argument('--participantIndependent', type=str, default='True')
    parser.add_argument('--totalDays', type=int, default=3)
    parser.add_argument('--task', type=str, default="classify")
    args = parser.parse_args()

    csv = read_csv("data/eureka_feat_20171122.csv")

    # finds the max and min for each feature
    maximum_features, minimum_features = model_utilities.find_max_min(csv)

    print("TOTAL DAYS: " + str(args.totalDays))
    #model_utilities.find_important_features(csv)




    # length of user id is 63
    user_ids = []
    for row in csv:
        if row[1] not in user_ids:
            user_ids = np.append(user_ids, row[1])

    if args.task == "classify":
        if args.model == "simpleNN":
            if args.participantIndependent == 'True':
                total_simpleNN_independent_acc = 0
                for i in range(0, 5):
                    simpleNN_independent_preds, simpleNN_independent_truths = simpleNN_classify.simple_NN_classification_participant_independent(
                        csv,
                        maximum_features,
                        minimum_features,
                        args.totalDays
                    )
                    simpleNN_independent_acc = metrics.calculate_accuracy(
                        simpleNN_independent_preds,
                        simpleNN_independent_truths)
                    metrics.confusionMetric(simpleNN_independent_preds,simpleNN_independent_truths)

                    total_simpleNN_independent_acc += simpleNN_independent_acc

                print("AVERAGE SIMPLE NN independent Accuracy")
                print(total_simpleNN_independent_acc / float(5))

            else:
                total_simpleNN_dependent_acc = 0
                for i in range(0, 5):
                    # randomize the userids
                    indices = np.random.permutation(len(user_ids))
                    simpleNN_dependent_preds, simpleNN_dependent_truths = simpleNN_classify.simple_NN_classification_participant_dependent(
                        csv,
                        user_ids[indices[:38]],
                        user_ids[indices[38:51]],
                        user_ids[indices[51:]],
                        maximum_features,
                        minimum_features,
                        args.totalDays
                    )
                    simpleNN_dependent_acc = metrics.calculate_accuracy(simpleNN_dependent_preds, simpleNN_dependent_truths)
                    metrics.confusionMetric(simpleNN_dependent_preds, simpleNN_dependent_truths)

                    total_simpleNN_dependent_acc += simpleNN_dependent_acc

                print("AVERAGE SIMPLE NN dependent Accuracy")
                print(total_simpleNN_dependent_acc / float(5))

        elif args.model == "lstm":
            if args.participantIndependent == 'True':
                print("START LSTM INDEPENDENT CLASSIFY")
                total_LSTM_independent = 0
                for i in range(0, 5):
                    LSTM_independent_preds, LSTM_independent_truths = LSTM_classify.LSTM_classification_participant_independent(
                        csv,
                        maximum_features,
                        minimum_features,
                        args.totalDays
                    )
                    LSTM_independent_acc = metrics.calculate_accuracy(LSTM_independent_preds, LSTM_independent_truths)
                    metrics.confusionMetric(LSTM_independent_preds, LSTM_independent_truths)

                    total_LSTM_independent += LSTM_independent_acc

                print("AVERAGE LSTM independent")
                print(total_LSTM_independent / float(5))

            else:
                print("START LSTM DEPENDENT CLASSIFY")
                total_LSTM_dependent = 0
                for i in range(0, 5):
                    indices = np.random.permutation(len(user_ids))

                    LSTM_dependent_preds, LSTM_dependent_truths = LSTM_classify.LSTM_classification_participant_dependent(
                        csv,
                        user_ids[indices[:38]],
                        user_ids[indices[38:51]],
                        user_ids[indices[51:]],
                        maximum_features,
                        minimum_features,
                        args.totalDays
                    )
                    LSTM_dependent_acc = metrics.calculate_accuracy(LSTM_dependent_preds, LSTM_dependent_truths)
                    metrics.confusionMetric(LSTM_dependent_preds, LSTM_dependent_truths)

                    total_LSTM_dependent += LSTM_dependent_acc

                print("AVERAGE LSTM dependent")
                print(total_LSTM_dependent / float(5))

        elif args.model == "svm":
            if args.participantIndependent == 'True':
                print("START SVM INDEPENDENT")
                svm_preds, svm_truths = SVM_classify.classification_participant_independent(
                    csv,
                    maximum_features,
                    minimum_features,
                    args.totalDays
                )
                metrics.calculate_accuracy(svm_preds, svm_truths)
                metrics.confusionMetric(svm_preds, svm_truths)
            else:
                print("START SVM DEPENDENT")
                indices = np.random.permutation(len(user_ids))
                svm_preds, svm_truths = SVM_classify.classification_participant_dependent(
                    csv,
                    user_ids[indices[:38]],
                    user_ids[indices[38:51]],
                    user_ids[indices[51:]],
                    maximum_features,
                    minimum_features,
                    args.totalDays
                )
                metrics.calculate_accuracy(svm_preds, svm_truths)
                metrics.confusionMetric(svm_preds, svm_truths)



    elif args.task == "regression":
        if args.model == "lstm":
            if args.participantIndependent == 'True':
                print("START LSTM INDEPENDENT REGRESSION")
                total_LSTM_independent = 0
                for i in range(0, 5):
                    LSTM_independent_preds, LSTM_independent_truths = LSTM_regression.LSTM_regression_participant_independent(
                        csv,
                        maximum_features,
                        minimum_features,
                        args.totalDays
                    )
                    lstm_independent_mse = metrics.calculate_mean_squared_error(
                        LSTM_independent_preds,
                        LSTM_independent_truths
                    )

                    total_LSTM_independent += lstm_independent_mse

                print("AVERAGE LSTM independent MSE: ")
                print(total_LSTM_independent / float(5))

            else:
                print("START LSTM DEPENDENT REGRESSION")
                total_LSTM_dependent = 0
                for i in range(0, 5):
                    indices = np.random.permutation(len(user_ids))

                    LSTM_dependent_preds, LSTM_dependent_truths = LSTM_regression.LSTM_regression_participant_dependent(
                        csv,
                        user_ids[indices[:38]],
                        user_ids[indices[38:51]],
                        user_ids[indices[51:]],
                        maximum_features,
                        minimum_features,
                        args.totalDays
                    )
                    lstm_dependent_mse = metrics.calculate_mean_squared_error(
                        LSTM_dependent_preds,
                        LSTM_dependent_truths
                    )

                    total_LSTM_dependent += lstm_dependent_mse

                print("AVERAGE LSTM dependent MSE: ")
                print(total_LSTM_dependent / float(5))




