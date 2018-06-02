import numpy as np
import datetime
from utilities import normalize_utilities
from utilities import csv_reader

# gets csv, userids, today's date, and normalization features
def initialize(normalize_type):
    print("INITIALIZATION RUNNING")

    csv = csv_reader.read_csv("data/eureka_feat_20171122.csv")

    # finds the max and min for each feature
    maximum_features, minimum_features = normalize_utilities.find_max_min(csv)
    # finds the mean and standard dev for each feature
    std_features, mean_features = normalize_utilities.find_mean_std(csv)

    if normalize_type == "z-score":
        normalizer1 = mean_features
        normalizer2 = std_features
    elif normalize_type == "min-max":
        normalizer1 = minimum_features
        normalizer2 = maximum_features

    today = str(datetime.date.today())

    user_ids = []
    for row in csv:
        if row[1] not in user_ids:
            user_ids = np.append(user_ids, row[1])

    return csv, user_ids, today, normalizer1, normalizer2