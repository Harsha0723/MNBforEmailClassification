from itertools import count
import os
import re
import nltk 
from nltk.corpus import stopwords
import pandas as pd
import copy
from collections import Counter

def mergeDataset(dataset, isTrain):
    dirName = os.path.dirname(__file__)
   
    hamFiles = []
    spamFiles = []
    
    # Get the path of dataset directory
    pathOfClassFiles = os.path.join(dirName, dataset)
    
    # Append train or test path
    if isTrain:
        pathOfClassFiles = os.path.join(pathOfClassFiles, "train")
    else:
        pathOfClassFiles = os.path.join(pathOfClassFiles, "test")
    
    # Append class names to the current path
    pathOfHamEmails = os.path.join(pathOfClassFiles, "ham")
    pathOfSpamEmails = os.path.join(pathOfClassFiles, "spam")

    # Read from the path to files
    readFiles(hamFiles, pathOfHamEmails)
    readFiles(spamFiles, pathOfSpamEmails)

    # join all emails into a string, with space as a seperator
    allWords = " ".join(hamFiles + spamFiles)
    
    return hamFiles, spamFiles, allWords

def readFiles(classFilesList, path):
    os.chdir(path)

    # Gather text files in path 
    # open text files for read
    # append the file content to a list
    for file in os.listdir():
        if file.endswith(".txt"):
            filePath = os.path.join(path, file)
            try:
                with open(filePath, 'r') as f:
                    classFilesList.append(f.read())
            except:
                pass

def constructVocab(allWords):
    vocabulary = {}
    fileWords = re.findall("[a-zA-Z]+", allWords)
    # Remove stop words in english language
    stopWords = set(stopwords.words('english'))
    
    # Construct vocab
    for word in fileWords:
        word = word.lower()
        if word in vocabulary:
            continue
        else:
            if word not in stopWords:
                vocabulary[word] = 0
        
    return vocabulary

def getBernoulliModel(vocab, emailFiles):
    totalEmails = []
    classDict = {}

    # Get the words in each file
    for email in emailFiles:
        currentDict = copy.deepcopy(vocab)
        emailWords = re.findall("[a-zA-Z]+", email)   
        for word in emailWords:
            word = word.lower()
            
            if word in currentDict:
                currentDict[word] = 1
                classDict[word] = 1
       
        totalEmails.append(currentDict)
    
    dataFrame = pd.DataFrame(totalEmails)
    dataFrame['emails'] = emailFiles
    firstColumn = dataFrame.pop('emails')
    dataFrame.insert(0, 'emails', firstColumn)
    return dataFrame, totalEmails, classDict

def constructBernoulliModel(dataset, isTrain):
    hamFiles, spamFiles, allWords = mergeDataset(dataset, isTrain)
    
    vocab = constructVocab(allWords)
    
    # Get bernoulli model for each class
    hamModel, hamFiles, hamDict = getBernoulliModel(vocab, hamFiles)
    spamModel, spamFiles, spamDict = getBernoulliModel(vocab, spamFiles)
    
    # print(hamModel)
    # print(spamModel)
    return hamFiles, hamDict, spamFiles, spamDict, vocab, hamModel, spamModel

def getBoWModel(vocab, emailFiles):
    totalEmailsInBowRep = []
    # Aggregate count of each word in vocab in all emails
    classDict = {}

    # Get the words in each file
    for email in emailFiles:
        currentDict = copy.deepcopy(vocab)
        emailWords = re.findall("[a-zA-Z]+", email)

        for word in emailWords:
            word = word.lower();
            if word in currentDict:
                currentDict[word] += 1;
        # add prev frequency with current frequency         
        classDict = Counter(classDict) + Counter(currentDict)  
        totalEmailsInBowRep.append(currentDict)

    dataFrame = pd.DataFrame(totalEmailsInBowRep)
    dataFrame['emails'] = emailFiles
    firstColumn = dataFrame.pop('emails')
    dataFrame.insert(0, 'emails', firstColumn)
    return dataFrame, totalEmailsInBowRep, classDict

def constructBagOfWords(dataset, isTrain):
    hamFiles, spamFiles, allWords = mergeDataset(dataset, isTrain) 
    
    vocabulary = constructVocab(allWords)
    
    hamModel, totalHamFiles, hamDict = getBoWModel(vocabulary, hamFiles)
    spamModel, totalSpamFiles, spamDict = getBoWModel(vocabulary, spamFiles)
    
    print(hamModel)
    print(spamModel)
    
    return totalHamFiles, hamDict, totalSpamFiles, spamDict, vocabulary, hamModel, spamModel

constructBagOfWords("enron2", True)