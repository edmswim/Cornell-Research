import numpy as np
from metric import metrics
from utilities import file_utilities
from model.simpleNN import simpleNN_regression
from model.lstm import LSTM_regression
import run_initialization

def run(participantDependent, leave_one_patient, totalDays, normalize_type, model):
    csv, user_ids, today, normalizer1, normalizer2 = run_initialization.initialize(normalize_type)

    list_mse = []

    if participantDependent == 'True':
        print("START DEPENDENT REGRESSION")

        if leave_one_patient == 'False':
            path = "output/logs/regression/participant_dependent/" + today +   "/" + model + "/" + str(totalDays - 1) + "_previous_days/"
            file_utilities.create_folder(path)
            version_number = file_utilities.find_next_version_file(path)
            log_file = open(path + model + "_" + str(version_number) + ".csv", "w+")
            log_file.write("iteration,mse,total_samples,average_mse,standard_deviation\n")

            for i in range(0, 5):
                if model == "lstm":
                    print("lstm iteration: " + str(i))
                    dependent_preds, dependent_truths = LSTM_regression.regression_participant_dependent(
                        csv,
                        normalize_type,
                        normalizer1,
                        normalizer2,
                        totalDays,
                        None
                    )
                elif model == "simpleNN":
                    print("simpleNN iteration: " + str(i))
                    dependent_preds, dependent_truths = simpleNN_regression.regression_participant_dependent(
                        csv,
                        normalize_type,
                        normalizer1,
                        normalizer2,
                        totalDays,
                        None
                    )

                dependent_mse = metrics.calculate_mean_squared_error(
                    dependent_preds,
                    dependent_truths
                )

                list_mse = np.append(list_mse, dependent_mse)
                log_file.write(str(i)+","+str(dependent_mse)+"," +  \
                    str(len(dependent_preds))+","+ str(np.sum(list_mse) / float(i+1)) + \
                    "," + str(np.std(list_mse)) + "\n")

            log_file.close()

        else:

            for i in range(0, 5):
                leave_one_patient_id = np.random.randint(64)
                print("\nLEAVE ONE PATIENT OUT: " + str(user_ids[leave_one_patient_id]))

                if model == "lstm":
                    print("lstm iteration: " + str(i))
                    dependent_preds, dependent_truths = LSTM_regression.regression_participant_dependent(
                        csv,
                        normalize_type,
                        normalizer1,
                        normalizer2,
                        totalDays,
                        user_ids[leave_one_patient_id]
                    )
                elif model == "simpleNN":
                    print("simpleNN iteration: " + str(i))
                    dependent_preds, dependent_truths = simpleNN_regression.regression_participant_dependent(
                        csv,
                        normalize_type,
                        normalizer1,
                        normalizer2,
                        totalDays,
                        user_ids[leave_one_patient_id]
                    )

                dependent_mse = metrics.calculate_mean_squared_error(
                    dependent_preds,
                    dependent_truths
                )

                list_mse = np.append(list_mse, simpleNN_dependent_mse)

            print("Average MSE: " +  str(np.mean(list_mse)))
            print("Average standard deviation: " + str(np.std(list_mse)))




    else:
        print("START INDEPENDENT REGRESSION")

        if leave_one_patient == 'False':

            path = "output/logs/regression/participant_independent/" + today +   "/" + model + "/" + str(totalDays - 1) + "_previous_days/"
            file_utilities.create_folder(path)
            version_number = file_utilities.find_next_version_file(path)
            log_file = open(path + model + "_" + str(version_number) + ".csv", "w+")
            log_file.write("iteration,mse,total_samples,average_mse,standard_deviation\n")
            list_mse = []
            for i in range(0, 5):
                indices = np.random.permutation(len(user_ids))

                if model == "lstm":
                    print("lstm iteration: " + str(i))
                    independent_preds, independent_truths = LSTM_regression.regression_participant_independent(
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
                    print("simpleNN iteration: " + str(i))
                    independent_preds, independent_truths = simpleNN_regression.regression_participant_independent(
                        csv,
                        user_ids[indices[:38]],
                        user_ids[indices[38:51]],
                        user_ids[indices[51:]],
                        normalize_type,
                        normalizer1,
                        normalizer2,
                        totalDays
                    )

                independent_mse = metrics.calculate_mean_squared_error(
                    independent_preds,
                    independent_truths
                )

                list_mse = np.append(list_mse, independent_mse)

                log_file.write(str(i)+","+str(independent_mse)+"," +  \
                    str(len(independent_preds))+","+ str(np.sum(list_mse) / float(i+1)) + \
                    "," + str(np.std(list_mse)) + "\n")

            log_file.close()

        else:
            indices = np.random.permutation(len(user_ids))

            for i in range(0, 5):
                leave_one_patient_id = np.random.randint(63)
                print("\nLEAVE ONE PATIENT OUT: " + str(user_ids[indices[leave_one_patient_id]]))

                if leave_one_patient_id + 1 < 63:
                    training_ids = np.append(user_ids[indices[:leave_one_patient_id]], user_ids[indices[leave_one_patient_id + 1:]])
                else:
                    training_ids = user_ids[indices[:leave_one_patient_id]]

                if model == "lstm":
                    print("lstm iteration: " + str(i))
                    independent_preds, independent_truths = LSTM_regression.regression_participant_independent(
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
                    print("simpleNN iteration: " + str(i))
                    independent_preds, independent_truths = simpleNN_regression.regression_participant_independent(
                        csv,
                        training_ids,
                        [],
                        user_ids[indices[leave_one_patient_id]],
                        normalize_type,
                        normalizer1,
                        normalizer2,
                        totalDays
                    )


                independent_mse = metrics.calculate_mean_squared_error(
                    independent_preds,
                    independent_truths
                )

                list_mse = np.append(list_mse, independent_mse)

            print("Average MSE: " +  str(np.mean(list_mse)))
            print("Average standard deviation: " + str(np.std(list_mse)))
