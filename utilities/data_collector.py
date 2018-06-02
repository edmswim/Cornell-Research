import numpy as np

# collects features of current day and previous days
# index represents the current day
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


#check if the same user is across all the days data
def isSameUserAcross(daysData):
    userid = daysData[0][1]
    for i in range(1, len(daysData)):
        if daysData[i][1] != userid:
            return False
    return True




