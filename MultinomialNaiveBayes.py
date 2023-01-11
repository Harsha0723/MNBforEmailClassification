# Multinomial Navie Bayes using Bag of Words model to represent data
import ConstructDataModels
from math import log10 as log
import EvaluationMetrics
from decimal import Decimal

def evaluateMNB(dataset):
    # Construct a bag of words model of train set
    hamTrainEmails, hamTrainDict, spamTrainEmails, spamTrainDict, trainVocabulary, hM, sM = ConstructDataModels.constructBagOfWords(dataset, True)
    
    # Train Multinomial Naive Baye's classifier
    classPrior, conditionalProb, laplaceSmooth = trainMNB(hamTrainEmails, hamTrainDict, spamTrainEmails, spamTrainDict, trainVocabulary)

    # Test the trained classifier
    hamTestEmails, hamTestDict, spamTestEmails, spamTestDict, testVocabulary, hM, sM = ConstructDataModels.constructBagOfWords(dataset, False)

    predictedHam = []
    for hamTestSample in hamTestEmails:
        predictedHam.append(testMNB(classPrior, conditionalProb, laplaceSmooth, hamTestSample))
    
    actualHam = [0] * len(hamTestEmails)
   
    predictedSpam = []
    for spamTestSample in spamTestEmails:
        predictedSpam.append(testMNB(classPrior, conditionalProb, laplaceSmooth, spamTestSample))

    actualSpam = [1] * len(spamTestEmails)

    expected = actualHam + actualSpam
    result = predictedHam + predictedSpam

    accuracy = EvaluationMetrics.accuracy(expected, result)
    precision = EvaluationMetrics.precision(expected, result)
    recall = EvaluationMetrics.recall(expected, result)
    f1Score = EvaluationMetrics.f1Score(recall, precision)

    print(dataset)
    print("Accuaracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1 Score: ", f1Score)

def trainMNB(hamEmails, hamDict, spamEmails, spamDict, vocab): 
    N = float(len(hamEmails) + len(spamEmails))

    conditionalProb = {}
    conditionalProb["ham"] = {}
    conditionalProb["spam"] = {}
    
    # class prior = Nc/N
    classPrior = {}
    classPrior["ham"] = log(Decimal(len(hamEmails) / N))
    classPrior["spam"] = log(Decimal(len(spamEmails) / N))

    laplaceSmooth = {}
    laplaceSmooth["spam"] = {}
    laplaceSmooth["ham"] = {}
    
    # Gather all the words in spam and ham classes
    # Using 1-Laplace Smoothing
    # P[X|y] = (P(Xi|y)^k + 1) / (Vocab of all files + vocab of class)
    totalHamWords = sum(hamDict.values())
    totalSpamWords = sum(spamDict.values())
    totalWords = len(vocab)

    for word in list(hamDict):
        conditionalProb["ham"][word] = log((1 + hamDict[word])/ float(totalWords + totalHamWords))

    for word in list(spamDict):
        conditionalProb["spam"][word] = log((1 + spamDict[word])/ float(totalWords + totalSpamWords))

    laplaceSmooth["spam"] = log(1/(float(totalWords + totalSpamWords)))
    laplaceSmooth["ham"] = log(1/(float(totalWords + totalHamWords)))
    
    return classPrior, conditionalProb, laplaceSmooth

def testMNB(classPrior, conditionalProb, laplaceSmooth, testSample):
    finalProb = {}
    # argmax log(P(x|Y)) + log(P(Y))
    for c in list(classPrior):
        finalProb[c] = classPrior[c]
        for word in list(testSample):
            if testSample[word] != 0:
                try:
                    finalProb[c] += conditionalProb[c][word]
                except KeyError:
                    finalProb[c] += laplaceSmooth[c]
    
    if finalProb["spam"] > finalProb["ham"]:
        return 1
    else:
        return 0

#evaluateMNB("enron2")

#evaluateMNB("enrol1")

#evaluateMNB("enrol")
