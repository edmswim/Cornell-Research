import numpy as np
import math


# finds maximum and minimum of features across the entire dataset (column wise)
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



# finds mean and standard deviation of features across the entire dataset (column wise)
def find_mean_std(csv):
    ema_scores = [93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106]
    csv = np.array(csv)
    std = np.array([0, 0, 0])
    mean = np.array([0, 0, 0])

    for i in range(3, csv.shape[1]):
        # if it's ema scores, just convert to 0 because we don't care since we aren't normalizing this
        if i in ema_scores:
            std = np.append(std, 0)
            mean = np.append(mean, 0)
        else:
            arr = np.array([row[i] for row in csv])
            num_arr = np.array([0.0 if math.isnan(float(numeric_string)) else float(numeric_string) for numeric_string in arr])
            std = np.append(std, np.std(num_arr, axis=0))
            mean = np.append(mean, np.mean(num_arr, axis=0))
    return std, mean