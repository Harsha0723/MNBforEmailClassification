from copy import copy
import ConstructDataModels
import random
import copy
import EvaluationMetrics
import numpy as np

def bernoulliEvaluateMCAPLR(dataset):
    hamTrainEmails, hamVocab, spamTrainEmails, spamVocab, trainVocabulary, hM, sM = ConstructDataModels.constructBernoulliModel(dataset, True)
    
    trainDataset, validationDataset = splitDataset(hamTrainEmails, spamTrainEmails)
    learnedLambda = learnLambda(trainDataset, validationDataset, trainVocabulary)
    
    trainingData = trainDataset + validationDataset
    modelWeights = trainMCAPLR(trainingData, learnedLambda, 0.01, 200, trainVocabulary)
    
    hamTestEmails, hamTestVocab, spamTestEmails, spamTestVocab, testVocabulary, hM, sM = ConstructDataModels.constructBernoulliModel(dataset, False)
    
    predictedHam = []
    for hamTestSample in hamTestEmails:
        predictedHam.append(testMCAPLR(hamTestSample, modelWeights))
    
    actualHam = [0] * len(hamTestEmails)
   
    predictedSpam = []
    for spamTestSample in spamTestEmails:
        predictedSpam.append(testMCAPLR(spamTestSample, modelWeights))

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

def bowEvaluateMCAPLR(dataset):
    hamTrainEmails, hamTrainDict, spamTrainEmails, spamTrainDict, vocabulary, hM, sM = ConstructDataModels.constructBagOfWords(dataset, True)
    
    trainDataset, validationDataset = splitDataset(hamTrainEmails, spamTrainEmails)
    learnedLambda = learnLambda(trainDataset, validationDataset, vocabulary)

    training_data = trainDataset + validationDataset
    modelWeights = trainMCAPLR(training_data, learnedLambda, 0.01, 200, vocabulary)
    
    hamTestEmails, hamTestDict, spamTestEmails, spamTestDict, testVocabulary, hM, sM = ConstructDataModels.constructBagOfWords(dataset, False)

    predictedHam = []
    for hamTestSample in hamTestEmails:
        predictedHam.append(testMCAPLR(hamTestSample, modelWeights))
    
    actualHam = [0] * len(hamTestEmails)
   
    predictedSpam = []
    for spamTestSample in spamTestEmails:
        predictedSpam.append(testMCAPLR(spamTestSample, modelWeights))

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

def splitDataset(hamEmails, spamEmails):
    # appending class value to data model 
    for dict in spamEmails:
        dict["classLabel"] = 1
        dict["w0"] = 1
    for dict in hamEmails:
        dict["classLabel"] = 0
        dict["w0"] = 1
    combine_email = spamEmails + hamEmails
    random.seed(4)
    random.shuffle(combine_email)
    
    seventyPercent = int(len(combine_email) * 0.70)
    trainFrame, validateFrame = combine_email[0:seventyPercent], combine_email[seventyPercent:-1]
    
    return trainFrame, validateFrame

def getPosteriorValue(modelWeights, dataSample):
    result = modelWeights['w0'] * 1
    
    # Calculating z = wi * xi
    for feature in dataSample:
        if feature == 'class_spam_ham' or feature == 'w0':
            continue
        else:
            if feature in modelWeights and feature in dataSample:
                result += (modelWeights[feature] * dataSample[feature])
            
    return 1/(float(1+np.exp(-result)))

def trainMCAPLR(trainDataset, lambda_, learningRate, iter_, vocab):
    # Weight / Parameters of model - should be of length vocab.
    modelWeights = copy.deepcopy(vocab)

    # Initialise all weights to 0
    for weight in modelWeights:
        modelWeights[weight] = 0
    
    modelWeights['w0'] = 0
    # for all data samples
        # wi = wi + learningRate * (Xi) *(class of sample - posteriorValue)
    # Add regularization term which is learningRate * lambra * wi
    # Xi - dataSample[feature]
    # Yi - class of dataSample
    # posteriorValue = P(Yi=1|Xi)
    for currentIter in range(iter_):
        for dataSample in trainDataset:
            posteriorValue = getPosteriorValue(modelWeights, dataSample)

            param_sum = 0
            for feature in vocab:
                # update only if Xi != 0
                if dataSample[feature] != 0:
                    if feature == 'w0':
                        param_sum += learningRate * (dataSample['class_spam_ham'] - posteriorValue)
                    else:
                        param_sum += learningRate * (dataSample[feature] * (dataSample['class_spam_ham'] - posteriorValue))

                modelWeights[feature] += param_sum - learningRate * lambda_ * modelWeights[feature]
    
    return modelWeights

def learnLambda(trainDataset, validationDataset, vocab):
    learningRate = 0.01
    accuracy = 0.0
    finalLambda = 2
    lenValidation = len(validationDataset)
    # For each value of lambda, generate a model on trainDataset
    # Choose the lambda, that gives maximum accuracy on validationDataset
    for lambda_ in range(1,10,2):
        modelWeights = trainMCAPLR(trainDataset, lambda_, learningRate, 25, vocab)

        correctPredict = 0
        for document in validationDataset:
            testOutput = testMCAPLR(document, modelWeights)
            
            if testOutput == document['classLabel']:
                correctPredict += 1
                
        temp_accuracy = correctPredict / float(lenValidation)
        if temp_accuracy > accuracy:
            accuracy = temp_accuracy
            finalLambda = lambda_
                           
    return finalLambda
    
def testMCAPLR(testSample, modelWeights):
    # sigma(Wi * Xi) > 0, then spam
    result = modelWeights['w0'] * 1
    
    for feature in testSample:
        if feature =='w0' or feature =='classLabel':
            continue
        else:
            if feature in modelWeights and feature in testSample:
                result += (modelWeights[feature] * testSample[feature])
            
    if result < 0:
        return 0
    else:
        return 1
    
#bernoulliEvaluateMCAPLR("enron")
#bernoulliEvaluateMCAPLR("enron1")
#bernoulliEvaluateMCAPLR("enron2")

#bowEvaluateMCAPLR("enron")
#bowEvaluateMCAPLR("enron1")
bowEvaluateMCAPLR("enron2")

