import numpy as np
import math
from sklearn.model_selection import train_test_split

class GaussianBayes:

    def fit(self, X, Y, test_size=0.5):
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=0)
        self.summaries = self.summarizeByClass(X_train, y_train)

    def separateByClass(self, features, classes):
        separated = {}
        for i in range(len(classes)):
            vector = features[i]
            if (classes[i] not in separated):
                separated[classes[i]] = []
            separated[classes[i]].append(vector)
        return separated

    def mean(self, numbers):
        return np.mean(numbers)

    def stdev(self, numbers):
        return np.std(numbers)

    def summarize(self, dataset):
        summaries = [(self.mean(attribute), self.stdev(attribute)) for attribute in zip(*dataset)]
        return summaries

    def summarizeByClass(self, features, classes):
        separated = self.separateByClass(features, classes)
        summaries = {}
        for classValue, instances in separated.items():
            summaries[classValue] = self.summarize(instances)
        return summaries

    def calculateProbability(self, x, mean, stdev):
        exponent = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(stdev, 2))))
        return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent

    def calculateClassProbabilities(self, summaries, inputVector):
        probabilities = {}
        for classValue, classSummaries in summaries.items():
            probabilities[classValue] = 1
            for i in range(len(classSummaries)):
                mean, stdev = classSummaries[i]
                x = inputVector[i]
                probabilities[classValue] *= self.calculateProbability(x, mean, stdev)
        return probabilities

    def getPredicton(self, summaries, inputVector):
        probabilities = self.calculateClassProbabilities(summaries, inputVector)
        bestLabel, bestProb = None, -1
        for classValue, probability in probabilities.items():
            if bestLabel is None or probability > bestProb:
                bestProb = probability
                bestLabel = classValue
        return bestLabel

    def getPredictions(self, summaries, testSet):
        predictions = []
        for i in range(len(testSet)):
            result = self.getPredicton(summaries, testSet[i])
            predictions.append(result)
        return predictions

    def accuracy(self, predictions, testSet):
        correct = 0
        for i in range(len(testSet)):
            if testSet[i] == predictions[i]:
                correct += 1
        return (correct / float(len(testSet))) * 100.0

    def predict(self, X):
        return self.getPredictions(self.summaries, X)