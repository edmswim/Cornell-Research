import numpy as np
from metric import metrics
from utilities import file_utilities

from model.lstm import LSTM_classify
from model.simpleNN import simpleNN_classify
from model.svm import SVM_classify
import run_initialization


def run(participantDependent, leave_one_patient, totalDays, normalize_type, model):
    csv, user_ids, today, normalizer1, normalizer2 = run_initialization.initialize(normalize_type)
    list_acc = []

    if participantDependent == 'True':
        print("START DEPENDENT CLASSIFY")

        if leave_one_patient == 'False':
            path = "output/logs/classification/participant_dependent/" + today +   "/" + model + "/" + str(totalDays - 1) + "_previous_days/"
            file_utilities.create_folder(path)
            version_number = file_utilities.find_next_version_file(path)
            log_file = open(path + model + "_" + str(version_number) + ".csv", "w+")
            log_file.write("iteration,accuracy,total_samples,average_acc,standard_deviation\n")

            for i in range(0, 5):

                if model == "lstm":
                    dependent_preds, dependent_truths = LSTM_classify.classify_participant_dependent(
                        csv,
                        normalize_type,
                        normalizer1,
                        normalizer2,
                        totalDays,
                        None
                    )
                elif model == "simpleNN":
                    dependent_preds, dependent_truths = simpleNN_classify.classification_participant_dependent(
                        csv,
                        normalize_type,
                        normalizer1,
                        normalizer2,
                        totalDays,
                        None
                    )
                elif model == "svm":
                    dependent_preds, dependent_truths = SVM_classify.classification_participant_dependent(
                        csv,
                        normalize_type,
                        normalizer1,
                        normalizer2,
                        totalDays,
                        None
                    )

                dependent_acc = metrics.calculate_accuracy(dependent_preds, dependent_truths)
                metrics.confusionMetric(dependent_preds, dependent_truths)

                list_acc = np.append(list_acc, dependent_acc)

                log_file.write(str(i)+","+str(dependent_acc)+"," +  \
                    str(len(dependent_preds))+","+ str(np.sum(list_acc) / float(i+1)) + \
                    "," + str(np.std(list_acc)) + "\n")

            log_file.close()

        else:
            for i in range(0, 5):
                leave_one_patient_id = np.random.randint(64)
                print("\nLEAVE ONE PATIENT OUT: " + str(user_ids[leave_one_patient_id]))

                if model == "lstm":
                    dependent_preds, dependent_truths = LSTM_classify.classify_participant_dependent(
                        csv,
                        normalize_type,
                        normalizer1,
                        normalizer2,
                        totalDays,
                        user_ids[leave_one_patient_id]
                    )

                elif model == "simpleNN":
                    dependent_preds, dependent_truths = simpleNN_classify.classification_participant_dependent(
                        csv,
                        normalize_type,
                        normalizer1,
                        normalizer2,
                        totalDays,
                        user_ids[leave_one_patient_id]
                    )
                elif model == "svm":
                    dependent_preds, dependent_truths = SVM_classify.classification_participant_dependent(
                        csv,
                        normalize_type,
                        normalizer1,
                        normalizer2,
                        totalDays,
                        user_ids[leave_one_patient_id]
                    )

                dependent_acc = metrics.calculate_accuracy(dependent_preds, dependent_truths)
                metrics.confusionMetric(dependent_preds, dependent_truths)
                list_acc = np.append(list_acc, dependent_acc)

            print("Average mean: " +  str(np.mean(list_acc)))
            print("Average standard deviation: " + str(np.std(list_acc)))


    else:
        print("START INDEPENDENT CLASSIFY")

        if leave_one_patient == 'False':
            path = "output/logs/classification/participant_independent/" + today + "/" + model + "/" + str(totalDays - 1) + "_previous_days/"
            file_utilities.create_folder(path)
            version_number = file_utilities.find_next_version_file(path)
            log_file = open(path + model + "_" + str(version_number) + ".csv", "w+")
            log_file.write("iteration,accuracy,total_samples,average_acc,standard_deviation\n")

            for i in range(0, 5):
                indices = np.random.permutation(len(user_ids))

                if model == "lstm":
                    independent_preds, independent_truths = LSTM_classify.classify_participant_independent(
                        csv,
                        user_ids[indices[:38]],
                        user_ids[indices[38:51]],
                        user_ids[indices[51:]],
                        normalize_type,
                        normalizer1,
                        normalizer2,
                        totalDays
                    )

                elif model == "simpleNN":
                    independent_preds, independent_truths = simpleNN_classify.classification_participant_independent(
                        csv,
                        user_ids[indices[:38]],
                        user_ids[indices[38:51]],
                        user_ids[indices[51:]],
                        normalize_type,
                        normalizer1,
                        normalizer2,
                        totalDays
                    )

                elif model == "svm":
                    independent_preds, independent_truths = SVM_classify.classification_participant_independent(
                        csv,
                        user_ids[indices[:38]],
                        user_ids[indices[38:51]],
                        user_ids[indices[51:]],
                        normalize_type,
                        normalizer1,
                        normalizer2,
                        totalDays
                    )

                independent_acc = metrics.calculate_accuracy(independent_preds, independent_truths)
                metrics.confusionMetric(independent_preds, independent_truths)
                list_acc = np.append(list_acc, independent_acc)

                log_file.write(str(i)+","+str(independent_acc)+"," + \
                    str(len(independent_preds))+","+ str(np.sum(list_acc) / float(i+1)) + \
                    "," + str(np.std(list_acc)) + "\n")

            log_file.close()

        else:

            for i in range(0, 5):
                indices = np.random.permutation(len(user_ids))

                # do leave one patient out
                leave_one_patient_id = np.random.randint(63)
                print("\nLEAVE ONE PATIENT OUT: " + str(user_ids[indices[leave_one_patient_id]]))

                if leave_one_patient_id + 1 < 63:
                    training_ids = np.append(user_ids[indices[:leave_one_patient_id]], user_ids[indices[leave_one_patient_id + 1:]])
                else:
                    training_ids = user_ids[indices[:leave_one_patient_id]]

                if model == "lstm":
                    independent_preds, independent_truths = LSTM_classify.classify_participant_independent(
                        csv,
                        training_ids,
                        [],
                        user_ids[indices[leave_one_patient_id]],
                        normalize_type,
                        normalizer1,
                        normalizer2,
                        totalDays
                    )
                elif model == "simpleNN":
                    independent_preds, independent_truths = simpleNN_classify.classification_participant_independent(
                        csv,
                        training_ids,
                        [],
                        user_ids[indices[leave_one_patient_id]],
                        normalize_type,
                        normalizer1,
                        normalizer2,
                        totalDays
                    )
                elif model == "svm":
                    independent_preds, independent_truths = SVM_classify.classification_participant_independent(
                        csv,
                        training_ids,
                        [],
                        user_ids[indices[leave_one_patient_id]],
                        normalize_type,
                        normalizer1,
                        normalizer2,
                        totalDays
                    )

                independent_acc = metrics.calculate_accuracy(independent_preds, independent_truths)
                metrics.confusionMetric(independent_preds, independent_truths)
                list_acc = np.append(list_acc, independent_acc)

            print("Average mean: " +  str(np.mean(list_acc)))
            print("Average standard deviation: " + str(np.std(list_acc)))

