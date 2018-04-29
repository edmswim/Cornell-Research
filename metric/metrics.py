import numpy as np

def confusionMetric(preds, truths):
    ret1 = np.zeros((4,4))
    ret2 = np.zeros((4,4))

    mapping = {}
    mapping[0] = np.zeros(4)
    mapping[1] = np.zeros(4)
    mapping[2] = np.zeros(4)
    mapping[3] = np.zeros(4)

    countZeros = 0
    countOnes = 0
    countTwos = 0
    countThrees = 0

    for i in range(0, len(preds)):
        if truths[i] == 0:
            countZeros += 1

        if truths[i] == 1:
            countOnes += 1

        if truths[i] == 2:
            countTwos += 1

        if truths[i] == 3:
            countThrees += 1

        arr = mapping[truths[i]]
        arr[int(preds[i])] += 1
        mapping[truths[i]] = arr

    print("Confusion Metric:")
    print('#0s: ' + str(countZeros))
    print('#1s: ' + str(countOnes))
    print('#2s: ' + str(countTwos))
    print('#3s: ' + str(countThrees))

    ret1[0] = mapping[0]
    ret1[1] = mapping[1]
    ret1[2] = mapping[2]
    ret1[3] = mapping[3]

    print(ret1)

    ret2[0] = mapping[0] / float(countZeros)
    ret2[1] = mapping[1] / float(countOnes)
    ret2[2] = mapping[2] / float(countTwos)
    ret2[3] = mapping[3] / float(countThrees)

    print(ret2)


#for classification task
def calculate_accuracy(preds, truths):
    print("Number of Predictions: " + str(len(preds)))
    numberCorrect = 0
    for i in range(0, len(preds)):
        if preds[i] == truths[i]:
            numberCorrect += 1
    print("Accuracy: " + str(numberCorrect / float(len(preds))))
    return (numberCorrect / float(len(preds)))


#for regression task
def calculate_mean_squared_error(preds, truths):
    mse = 0
    for i in range(0, len(preds)):
        diff = preds[i] - truths[i]
        mse += (diff * diff)

    print("Mean Squared Error: " + str(mse / float(len(preds))))
    return (mse / float(len(preds)))
