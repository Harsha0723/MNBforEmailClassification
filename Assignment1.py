import sys
import ConstructDataModels
import MultinomialNaiveBayes
import DiscreteNaiveBayes
import MCAP_LR
import SGDC

arguments = list(sys.argv)

dataset = arguments[1]
algorithm = arguments[2]

def runAlgo():

    mapping = {'mnb': 'Multinomial Naive Bayes', 'dnb': 'Discrete Naive Bayes', 'lr' : 'Logistic Regression', 'sgd': 'Stochastic Gradient Descent', 'bow':'Bag of Words', 'bern': 'Bernoulli'}

    try:
        model = arguments[3]
    except:
        pass

    if algorithm == 'mnb':
        MultinomialNaiveBayes.evaluateMNB(dataset)
    elif algorithm == 'dnb':
        DiscreteNaiveBayes.evaluateDNB(dataset)
    elif algorithm == 'lr':
        if model == 'bow':
            MCAP_LR.bowEvaluateMCAPLR(dataset)
        elif model == 'bern':
            MCAP_LR.bernoulliEvaluateMCAPLR(dataset)
        else:
            print("Incorrect Argument")
    elif algorithm == 'sgd':
        if model == 'bow':
            SGDC.evaluateSGDBOW(dataset)
        elif model == 'bern':
            SGDC.evaluateSGDBOW(dataset)
        else:
            print("Incorrect Argument")
    elif algorithm == 'matrix_bow':
        result = ConstructDataModels.constructBagOfWords(dataset, True)
        print("Train - Spam Matrix", result[7])
        print("Train - Ham Matirx", result[6])
        
        result = ConstructDataModels.constructBagOfWords(dataset, False)
        
        print("Test - Spam Matrix", result[7])
        print("Test - Ham Matrix", result[6])
    elif algorithm == 'matrix_bernoulli':
        result = ConstructDataModels.constructBernoulliModel(dataset, True)
        print("Train - Spam Matrix", result[7])
        print("Train - Ham Matirx", result[6])
        
        result = ConstructDataModels.constructBernoulliModel(dataset, False)
        
        print("Test - Spam Matrix", result[7])
        print("Test - Ham Matrix", result[6])

if __name__ == '__main__':
    runAlgo()

