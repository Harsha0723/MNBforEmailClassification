import ConstructDataModels
import random
import EvaluationMetrics
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV

def evaluateSGDBOW(dataset):
    hamTrainEmails, hamTrainDict, spamTrainEmails, spamTrainDict, vocabulary, hM, sM = ConstructDataModels.constructBagOfWords(dataset, True)
    
    hamTestEmails, hamTestDict, spamTestEmails, spamTestDict, testVocabulary, hM, sM = ConstructDataModels.constructBagOfWords(dataset, False)

    training_data, validation_data = splitData(spamTrainEmails, hamTrainEmails, True)
    
    testing_data = splitData(spamTestEmails, hamTestEmails, False)
    
    features = list(training_data[0])
    
    train_x_arr, train_y_arr = ConvertDataForSGD(training_data, features)
    test_x_arr, test_y_arr = ConvertDataForSGD(testing_data, features)
    validation_x_arr, validation_y_arr = ConvertDataForSGD(validation_data, features)
    
    classifier = tuneParams(validation_x_arr, validation_y_arr)
    
    #training the model
    trained_model = classifier.fit(train_x_arr, train_y_arr)
    
    #testing the model
    predicted_class = testSGD(trained_model, test_x_arr)
    
        
    accuracy = EvaluationMetrics.accuracy(test_y_arr, predicted_class)
    precision = EvaluationMetrics.precision(test_y_arr, predicted_class)
    recall = EvaluationMetrics.recall(test_y_arr, predicted_class)
    f1Score = EvaluationMetrics.f1Score(recall, precision)
    
    print(dataset)
    print(accuracy)
    print(precision)
    print(recall)
    print(f1Score)

    return accuracy, precision, recall, f1Score, predicted_class

def evaluateSGDBernoulli(dataset):
        hamTrainEmails, hamVocab, spamTrainEmails, spamVocab, trainVocabulary, hM, sM = ConstructDataModels.constructBernoulliModel(dataset, True)
   
        hamTestEmails, hamTestVocab, spamTestEmails, spamTestVocab, testVocabulary, hM, sM = ConstructDataModels.constructBernoulliModel(dataset, False)
    
        trainDataset, validationDataset = splitData(spamTrainEmails, hamTrainEmails, True)
        
        testDataset = splitData(spamTestEmails, hamTestEmails, False)
        
        features = list(trainDataset[0])
        
        trainX, trainY = ConvertDataForSGD(trainDataset, features)
        testX, testY = ConvertDataForSGD(testDataset, features)
        validationX, validationY = ConvertDataForSGD(validationDataset, features)
        
        classifier = tuneParams(validationX, validationY)
        
        #training the model
        trainedModel = classifier.fit(trainX, trainY)
        
        #testing the model
        predictedClass = testSGD(trainedModel, testX)
        
        accuracy = EvaluationMetrics.accuracy(testY, predictedClass)
        precision = EvaluationMetrics.precision(testY, predictedClass)
        recall = EvaluationMetrics.recall(testY, predictedClass)
        f1Score = EvaluationMetrics.f1Score(recall, precision)
        
        print(dataset)
        print("Accuaracy: ", accuracy)
        print("Precision: ", precision)
        print("Recall: ", recall)
        print("F1 Score: ", f1Score)

        return accuracy, precision, recall, f1Score

def splitData(spamEmails, hamEmails, isTrain):
    for dict in spamEmails:
        dict["classLabel"] = 1
    for dict in hamEmails:
        dict["classLabel"] = 0
          
    totalMails = spamEmails + hamEmails

    if not isTrain:
        return totalMails
    else:
        random.seed(4)
        random.shuffle(totalMails)
        
        trainSet, validateSet = totalMails[0:int(len(totalMails) * 0.70)], totalMails[int(len(totalMails)*0.70):-1]
        
        return trainSet, validateSet

def ConvertDataForSGD(dataSet, features):
    x, y = [], []
    
    for dataSample in dataSet:
        x_current_doc = []
        y.append(dataSample['classLabel'])
        
        for feature in features:
            try:
                x_current_doc.append(dataSample[feature])
            except:
                x_current_doc.append(0)
                
        x.append(x_current_doc)
        
    return x, y

def tuneParams(validationX, validationY):
    
    params = {'alpha' : (0.01, 0.05),
              'max_iter' : (range(500, 3000, 1000)),
              'learning_rate': ('optimal', 'invscaling', 'adaptive'),
              'eta0' : (0.3, 0.5),
              'tol' : (0.001, 0.005)}
    
    sgdClassifier = SGDClassifier()
    
    gridSearch = GridSearchCV(sgdClassifier, params, cv=5)
    gridSearch.fit(validationX, validationY)
    
    return gridSearch

def testSGD(classifier, testX):
    predictedClass = []
    
    for sample in testX:
        predictedClass.append(classifier.predict(np.reshape(sample, (1,-1))))
        
    return predictedClass

#evaluateSGDBernoulli("enron")
#evaluateSGDBernoulli("enron1")
#evaluateSGDBernoulli("enron2")

evaluateSGDBOW("enron")
evaluateSGDBOW("enron1")
evaluateSGDBOW("enron2")