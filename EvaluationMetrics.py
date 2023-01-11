# expectedY - dataset value
# actualY - predicted value

def accuracy(expectedY, actualY):
    accuracyCount = 0
    for each in range(len(expectedY)):
        if expectedY[each] == actualY[each]:
            accuracyCount+= 1
    return accuracyCount / float(len(expectedY))


def precision(expectedY, actualY):
    truePositives = 0
    falsePositives = 0
    for each in range(len(expectedY)):
        if expectedY[each] == actualY[each] and actualY[each] == 1:
            truePositives += 1
        if expectedY[each] != actualY[each] and actualY[each] == 1:
            falsePositives += 1
    return truePositives / float(truePositives + falsePositives)


def recall(expectedY, actualY):
    truePositives = 0
    falseNegatives = 0
    for each in range(len(expectedY)):
        if expectedY[each] == actualY[each] and actualY[each] == 1:
            truePositives += 1
        if expectedY[each] != actualY[each] and actualY[each] == 0:
            falseNegatives += 1
    return truePositives / float(truePositives + falseNegatives)


def f1Score(recall, precision): 
    return (2 * recall * precision) / float(recall + precision)
