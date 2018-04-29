import numpy as np
import csv
import math
import keras
import argparse
import random
import datetime

from model.simpleNN import simpleNN_classify
from model.lstm import LSTM_classify
from model.svm import SVM_classify

from model.lstm import LSTM_regression
from model.simpleNN import simpleNN_regression

from metric import metrics
from utilities import model_utilities
from utilities import file_utilities
from utilities import normalize_utilities


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
    parser.add_argument('--participantDependent', type=str, default='True')
    parser.add_argument('--totalDays', type=int, default=3)
    parser.add_argument('--task', type=str, default="classify")
    parser.add_argument('--normalize', type=str, default="z-score")
    args = parser.parse_args()

    csv = read_csv("data/eureka_feat_20171122.csv")

    # finds the max and min for each feature
    maximum_features, minimum_features = normalize_utilities.find_max_min(csv)
    # finds the mean and standard dev for each feature
    std_features, mean_features = normalize_utilities.find_mean_std(csv)

    print("TOTAL DAYS: " + str(args.totalDays))
    #model_utilities.find_important_features(csv)


    if args.normalize == "z-score":
        normalizer1 = mean_features
        normalizer2 = std_features
    elif args.normalize == "min-max":
        normalizer1 = minimum_features
        normalizer2 = maximum_features



    today = datetime.date.today()
    today = str(today)



    # length of user id is 63
    user_ids = []
    for row in csv:
        if row[1] not in user_ids:
            user_ids = np.append(user_ids, row[1])

    if args.task == "classify":
        if args.model == "simpleNN":
            if args.participantDependent == 'True':
                print("START SimpleNN DEPENDENT CLASSIFY")

                path = "output/logs/classification/participant_dependent/" + today +   "/simpleNN/" + str(args.totalDays - 1) + "_previous_days/"
                file_utilities.create_folder(path)
                version_number = file_utilities.find_next_version_file(path)
                log_file = open(path + "simpleNN_" + str(version_number) + ".csv", "w+")
                log_file.write("iteration,accuracy,total_samples,average_acc,standard_deviation\n")
                list_acc = []

                for i in range(0, 5):
                    simpleNN_dependent_preds, simpleNN_dependent_truths = simpleNN_classify.classification_participant_dependent(
                        csv,
                        args.normalize,
                        normalizer1,
                        normalizer2,
                        args.totalDays
                    )
                    simpleNN_dependent_acc = metrics.calculate_accuracy(
                        simpleNN_dependent_preds,
                        simpleNN_dependent_truths)
                    metrics.confusionMetric(simpleNN_dependent_preds, simpleNN_dependent_truths)

                    list_acc = np.append(list_acc, simpleNN_dependent_acc)

                    log_file.write(str(i)+","+str(simpleNN_dependent_acc)+"," +  \
                        str(len(simpleNN_dependent_preds))+","+ str(np.sum(list_acc) / float(i+1)) + \
                        "," + str(np.std(list_acc)) + "\n")

                log_file.close()

            else:
                print("START SimpleNN INDEPENDENT CLASSIFY")
                path = "output/logs/classification/participant_independent/" + today +   "/simpleNN/" + str(args.totalDays - 1) + "_previous_days/"
                file_utilities.create_folder(path)
                version_number = file_utilities.find_next_version_file(path)
                log_file = open(path + "simpleNN_" + str(version_number) + ".csv", "w+")
                log_file.write("iteration,accuracy,total_samples,average_acc,standard_deviation\n")
                list_acc = []

                for i in range(0, 5):
                    # randomize the userids
                    indices = np.random.permutation(len(user_ids))
                    simpleNN_independent_preds, simpleNN_independent_truths = simpleNN_classify.classification_participant_independent(
                        csv,
                        user_ids[indices[:38]],
                        user_ids[indices[38:51]],
                        user_ids[indices[51:]],
                        args.normalize,
                        normalizer1,
                        normalizer2,
                        args.totalDays
                    )
                    simpleNN_independent_acc = metrics.calculate_accuracy(simpleNN_independent_preds, simpleNN_independent_truths)
                    metrics.confusionMetric(simpleNN_independent_preds, simpleNN_independent_truths)
                    list_acc = np.append(list_acc, simpleNN_independent_acc)

                    log_file.write(str(i)+","+str(simpleNN_independent_acc)+"," +  \
                        str(len(simpleNN_independent_preds))+","+ str(np.sum(list_acc) / float(i+1)) + \
                        "," + str(np.std(list_acc)) + "\n")

                log_file.close()


        elif args.model == "lstm":
            if args.participantDependent == 'True':
                print("START LSTM DEPENDENT CLASSIFY")
                path = "output/logs/classification/participant_dependent/" + today +   "/lstm/" + str(args.totalDays - 1) + "_previous_days/"
                file_utilities.create_folder(path)
                version_number = file_utilities.find_next_version_file(path)
                log_file = open(path + "lstm_" + str(version_number) + ".csv", "w+")
                log_file.write("iteration,accuracy,total_samples,average_acc,standard_deviation\n")
                list_acc = []

                for i in range(0, 5):
                    LSTM_dependent_preds, LSTM_dependent_truths = LSTM_classify.classify_participant_dependent(
                        csv,
                        args.normalize,
                        normalizer1,
                        normalizer2,
                        args.totalDays
                    )
                    LSTM_dependent_acc = metrics.calculate_accuracy(LSTM_dependent_preds, LSTM_dependent_truths)
                    metrics.confusionMetric(LSTM_dependent_preds, LSTM_dependent_truths)

                    list_acc = np.append(list_acc, LSTM_dependent_acc)

                    log_file.write(str(i)+","+str(LSTM_dependent_acc)+"," +  \
                        str(len(LSTM_dependent_preds))+","+ str(np.sum(list_acc) / float(i+1)) + \
                        "," + str(np.std(list_acc)) + "\n")

                log_file.close()


            else:
                print("START LSTM INDEPENDENT CLASSIFY")
                path = "output/logs/classification/participant_independent/" + today +   "/lstm/" + str(args.totalDays - 1) + "_previous_days/"
                file_utilities.create_folder(path)
                version_number = file_utilities.find_next_version_file(path)
                log_file = open(path + "lstm_" + str(version_number) + ".csv", "w+")
                log_file.write("iteration,accuracy,total_samples,average_acc,standard_deviation\n")
                list_acc = []
                for i in range(0, 5):
                    indices = np.random.permutation(len(user_ids))

                    LSTM_independent_preds, LSTM_independent_truths = LSTM_classify.classify_participant_independent(
                        csv,
                        user_ids[indices[:38]],
                        user_ids[indices[38:51]],
                        user_ids[indices[51:]],
                        args.normalize,
                        normalizer1,
                        normalizer2,
                        args.totalDays
                    )
                    LSTM_independent_acc = metrics.calculate_accuracy(LSTM_independent_preds, LSTM_independent_truths)
                    metrics.confusionMetric(LSTM_independent_preds, LSTM_independent_truths)

                    list_acc = np.append(list_acc, LSTM_independent_acc)

                    log_file.write(str(i)+","+str(LSTM_independent_acc)+"," +  \
                        str(len(LSTM_independent_preds))+","+ str(np.sum(list_acc) / float(i+1)) + \
                        "," + str(np.std(list_acc)) + "\n")

                log_file.close()



        elif args.model == "svm":
            if args.participantDependent == 'True':
                print("START SVM DEPENDENT")
                svm_preds, svm_truths = SVM_classify.classification_participant_dependent(
                    csv,
                    "z-score",
                    normalizer1,
                    normalizer2,
                    args.totalDays
                )
                metrics.calculate_accuracy(svm_preds, svm_truths)
                metrics.confusionMetric(svm_preds, svm_truths)
            else:
                print("START SVM INDEPENDENT")
                indices = np.random.permutation(len(user_ids))
                svm_preds, svm_truths = SVM_classify.classification_participant_independent(
                    csv,
                    user_ids[indices[:38]],
                    user_ids[indices[38:51]],
                    user_ids[indices[51:]],
                    args.normalize,
                    normalizer1,
                    normalizer2,
                    args.totalDays
                )
                metrics.calculate_accuracy(svm_preds, svm_truths)
                metrics.confusionMetric(svm_preds, svm_truths)












    elif args.task == "regression":

        if args.model == "lstm":
            if args.participantDependent == 'True':
                print("START LSTM DEPENDENT REGRESSION")

                path = "output/logs/regression/participant_dependent/" + today +   "/lstm/" + str(args.totalDays - 1) + "_previous_days/"
                file_utilities.create_folder(path)
                version_number = file_utilities.find_next_version_file(path)
                log_file = open(path + "lstm_" + str(version_number) + ".csv", "w+")
                log_file.write("iteration,mse,total_samples,average_mse,standard_deviation\n")
                list_mse = []
                for i in range(0, 5):
                    LSTM_dependent_preds, LSTM_dependent_truths = LSTM_regression.regression_participant_dependent(
                        csv,
                        args.normalize,
                        normalizer1,
                        normalizer2,
                        args.totalDays
                    )
                    lstm_dependent_mse = metrics.calculate_mean_squared_error(
                        LSTM_dependent_preds,
                        LSTM_dependent_truths
                    )

                    list_mse = np.append(list_mse, lstm_dependent_mse)
                    log_file.write(str(i)+","+str(lstm_dependent_mse)+"," +  \
                        str(len(LSTM_dependent_preds))+","+ str(np.sum(list_mse) / float(i+1)) + \
                        "," + str(np.std(list_mse)) + "\n")

                log_file.close()


            else:
                print("START LSTM INDEPENDENT REGRESSION")

                path = "output/logs/regression/participant_independent/" + today +   "/lstm/" + str(args.totalDays - 1) + "_previous_days/"
                file_utilities.create_folder(path)
                version_number = file_utilities.find_next_version_file(path)
                log_file = open(path + "lstm_" + str(version_number) + ".csv", "w+")
                log_file.write("iteration,mse,total_samples,average_mse,standard_deviation\n")
                list_mse = []
                for i in range(0, 5):
                    indices = np.random.permutation(len(user_ids))

                    LSTM_independent_preds, LSTM_independent_truths = LSTM_regression.regression_participant_independent(
                        csv,
                        user_ids[indices[:38]],
                        user_ids[indices[38:51]],
                        user_ids[indices[51:]],
                        args.normalize,
                        normalizer1,
                        normalizer2,
                        args.totalDays
                    )
                    lstm_independent_mse = metrics.calculate_mean_squared_error(
                        LSTM_independent_preds,
                        LSTM_independent_truths
                    )

                    list_mse = np.append(list_mse, lstm_independent_mse)

                    log_file.write(str(i)+","+str(lstm_independent_mse)+"," +  \
                        str(len(LSTM_independent_preds))+","+ str(np.sum(list_mse) / float(i+1)) + \
                        "," + str(np.std(list_mse)) + "\n")

                log_file.close()




        if args.model == "simpleNN":
            if args.participantDependent == 'True':
                print("START SIMPLE_NN DEPENDENT REGRESSION")

                path = "output/logs/regression/participant_dependent/" + today + "/simpleNN/" + str(args.totalDays - 1) + "_previous_days/"
                file_utilities.create_folder(path)
                version_number = file_utilities.find_next_version_file(path)
                log_file = open(path + "lstm_" + str(version_number) + ".csv", "w+")
                log_file.write("iteration,mse,total_samples,average_mse,standard_deviation\n")
                list_mse = []
                for i in range(0, 5):
                    simpleNN_dependent_preds, simpleNN_dependent_truths = simpleNN_regression.regression_participant_ddependent(
                        csv,
                        args.normalize,
                        normalizer1,
                        normalizer2,
                        args.totalDays
                    )
                    simpleNN_dependent_mse = metrics.calculate_mean_squared_error(
                        simpleNN_dependent_preds,
                        simpleNN_dependent_truths
                    )

                    list_mse = np.append(list_mse, simpleNN_dependent_mse)

                    log_file.write(str(i)+","+str(simpleNN_dependent_mse)+"," +  \
                        str(len(simpleNN_dependent_preds))+","+ str(np.sum(list_mse) / float(i+1)) + \
                        "," + str(np.std(list_mse)) + "\n")

                log_file.close()


            else:
                print("START SIMPLE_NN INDEPENDENT REGRESSION")

                path = "output/logs/regression/participant_independent/" + today + "/simpleNN/" + str(args.totalDays - 1) + "_previous_days/"
                file_utilities.create_folder(path)
                version_number = file_utilities.find_next_version_file(path)
                log_file = open(path + "lstm_" + str(version_number) + ".csv", "w+")
                log_file.write("iteration,mse,total_samples,average_mse,standard_deviation\n")
                list_mse = []
                for i in range(0, 5):
                    # randomize the userids
                    indices = np.random.permutation(len(user_ids))
                    simpleNN_independent_preds, simpleNN_independent_truths = simpleNN_regression.regression_participant_independent(
                        csv,
                        user_ids[indices[:38]],
                        user_ids[indices[38:51]],
                        user_ids[indices[51:]],
                        args.normalize,
                        normalizer1,
                        normalizer2,
                        args.totalDays
                    )
                    simpleNN_independent_mse = metrics.calculate_mean_squared_error(
                        simpleNN_independent_preds,
                        simpleNN_independent_truths
                    )

                    list_mse = np.append(list_mse, simpleNN_independent_mse)

                    log_file.write(str(i)+","+str(simpleNN_independent_mse)+"," +  \
                        str(len(simpleNN_independent_preds))+","+ str(np.sum(list_mse) / float(i+1)) + \
                        "," + str(np.std(list_mse)) + "\n")

                log_file.close()



