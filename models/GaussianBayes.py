import numpy as np
import math
from sklearn.model_selection import train_test_split

class GaussianBayes:

    def fit(self, X, Y, test_size=0.5):
        a_values = [-0.1, -0.05, -0.02, -0.01, -0.005, 0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1]
        b_values = [-0.1, -0.05, -0.02, -0.01, -0.005, 0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1]
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=0)
        error_best, best_a, best_b, errors = self.modelSelection(X_train, X_test, y_train, y_test, a_values,b_values)
        self.p_y = self.estimate_a_priori(y_train)
        self.summaries = self.summarizeByClass(X_train, y_train, best_a, best_b)
        print(best_a)
        print(best_b)

    def modelSelection(self, Xtrain, Xval, Ytrain, Yval, aValues, bValues):
        errors = np.zeros((len(aValues), len(bValues)))

        error_best = float("inf")

        for a in range(len(aValues)):
            for b in range(len(bValues)):
                p_y_x = self.estimate_p_y_x(self.estimate_a_priori(Ytrain), self.summarizeByClass(Xtrain, Ytrain, aValues[a], bValues[b]), Xval)
                errors[a,b] = self.error_fun(p_y_x, Yval)
                if errors[a,b] < error_best:
                    error_best = errors[a,b]
                    bestA = aValues[a]
                    bestB = bValues[b]

        return error_best, bestA, bestB, errors

    def estimate_a_priori(self, yTrain):
        M = np.unique(yTrain).size
        N = len(yTrain)

        p_y = np.zeros((M,1))

        for a in range(M):
            for b in range(N):
                if yTrain[b] == a:
                    p_y[a] = p_y[a] + 1
            p_y[a] = p_y[a]/N

        return p_y

    def separateByClass(self, features, classes):
        separated = {}
        for i in range(len(classes)):
            vector = features[i]
            if (classes[i] not in separated):
                separated[classes[i]] = []
            separated[classes[i]].append(vector)
        return separated

    def mean(self, numbers):
        return np.mean(numbers)+0.02

    def stdev(self, numbers):
        return np.std(numbers)+0.01

    def summarize(self, dataset, a, b):
        summaries = [(self.mean(attribute) + a - 0.002, self.stdev(attribute) + b - 0.001) for attribute in zip(*dataset)]
        return summaries

    def summarizeByClass(self, features, classes, a, b):
        separated = self.separateByClass(features, classes)
        summaries = {}
        for classValue, instances in separated.items():
            summaries[classValue] = self.summarize(instances, a, b)
        return summaries

    def estimate_p_y_x(self, p_y, summaries, X):
        N = np.size(X, 0)
        M = len(p_y)

        p_y_x = np.zeros((N, M))

        iloczyn = 1
        sum = 0

        for a in range(N):
            for b in range(M):
                for c in range(np.size(X, 1)):
                    mean, stdev = summaries[float(b)][c]
                    x = X[a,c]
                    iloczyn *= self.calculateProbability(x, mean, stdev)
                iloczyn = iloczyn * p_y[b]
                sum = sum + iloczyn
                iloczyn = 1
            for b in range(M):
                for c in range(np.size(X, 1)):
                    mean, stdev = summaries[float(b)][c]
                    x = X[a, c]
                    iloczyn *= self.calculateProbability(x, mean, stdev)
                iloczyn = iloczyn * p_y[b]
                p_y_x[a, b] = iloczyn / float(sum)
                iloczyn = 1
            sum = 0
        return p_y_x

    def error_fun(self, p_y_x, Y):
        error_val = 0

        for i in range(len(Y)):
            max = 0
            for j in range(np.size(p_y_x, 1)):
                if p_y_x[i, max] <= p_y_x[i, j]:
                    max = j
            if max != Y[i]:
                error_val = error_val + 1

        error_val = error_val / len(Y)

        return error_val

    def accuracy(self, p_y_x, Y):
        return 1 - self.error_fun(p_y_x, Y)

    def predict(self, X):
        return self.estimate_p_y_x(self.p_y, self.summaries, X)