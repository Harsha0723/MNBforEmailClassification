import ConstructDataModels
from math import log10 as log
import EvaluationMetrics
from decimal import Decimal

def evaluateDNB(dataset):
    hamTrainEmails, hamDict, spamTrainEmails, spamDict, trainVocabulary, hM, sM = ConstructDataModels.constructBernoulliModel(dataset, True)
    
    classPrior, conditionalProb, laplaceSmooth = trainDNB(hamTrainEmails, hamDict, spamTrainEmails, spamDict)
    
    hamTestEmails, testHamDict, spamTestEmails, testSpamDict, testVocabulary, hM, sM = ConstructDataModels.constructBernoulliModel(dataset, False)

    predictedHam = []
    for hamTestSample in hamTestEmails:
        predictedHam.append(testDNB(classPrior, conditionalProb, laplaceSmooth, hamTestSample))
    
    actualHam = [0] * len(hamTestEmails)
   
    predictedSpam = []
    for spamTestSample in spamTestEmails:
        predictedSpam.append(testDNB(classPrior, conditionalProb, laplaceSmooth, spamTestSample))

    actualSpam = [1] * len(spamTestEmails)

    expected = actualHam + actualSpam
    result = predictedHam + predictedSpam

    accuracy = EvaluationMetrics.accuracy(expected, result)
    precision = EvaluationMetrics.precision(expected, result)
    recall = EvaluationMetrics.recall(expected, result)
    f1Score = EvaluationMetrics.f1Score(recall, precision)

    print(dataset)
    print("Accuaracy: %1", accuracy)
    print("Precision: %1", precision)
    print("Recall: %1", recall)
    print("F1 Score: %1", f1Score)

def trainDNB(hamEmails, hamDict, spamEmails, spamDict): 
    N = len(hamEmails) + len(spamEmails)
    # Assign class Priors and conditional probabilities and add-one laplace smoothing
    conditionalProb = {}
    conditionalProb["ham"] = {}
    conditionalProb["spam"] = {}
    
     # class prior = Nc/N
    classPrior = {}
    classPrior["ham"] = log(len(hamEmails) / float(N))
    classPrior["spam"] = log(len(spamEmails) / float(N))

    laplaceSmooth = {}
    laplaceSmooth["spam"] = {}
    laplaceSmooth["ham"] = {}
    
    # Gather all the words in spam and ham classes
    totalHamFiles = len(hamEmails)
    totalSpamFiles = len(spamEmails)
    
    for word in list(hamDict):
        conditionalProb["ham"][word] = log(1+hamDict[word]/(float(totalHamFiles+2)))

    for word in list(spamDict):
        conditionalProb["spam"][word] = log(1+spamDict[word]/(float(totalSpamFiles+2)))

    # 1 Laplace smoothing
    laplaceSmooth["spam"] = log(1/(float(2 + totalSpamFiles)))
    laplaceSmooth["ham"] = log(1/(float(2 + totalHamFiles)))
    
    return classPrior, conditionalProb, laplaceSmooth

def testDNB(classPrior, conditionalProb, laplaceSmooth, testSample):
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

#evaluateDNB("enron")
#evaluateDNB("enron1")
#evaluateDNB("enron2")

