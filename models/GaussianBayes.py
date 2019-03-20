import numpy as np
import math
from sklearn.model_selection import train_test_split

class GaussianBayes:

    def fit(self, X, Y, test_size=0.5):
        a_values = [-0.1, -0.05, -0.03, -0.02, -0.01, -0.005, -0.003, -0.002, 0, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2, 0.3, 0.5]
        b_values = [-0.1, -0.05, -0.03, -0.02, -0.01, -0.005, -0.003, -0.002, 0, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2, 0.3, 0.5]
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size)
        self.p_y = self.estimate_a_priori(y_train)
        error_best, best_a, best_b, errors = self.modelSelection(self.p_y, X_train, X_test, y_train, y_test, a_values, b_values)
        self.summaries = self.summarizeByClass(X_train, y_train, best_a, best_b);

    def modelSelection(self, p_y, Xtrain, Xval, Ytrain, Yval, aValues, bValues):
        errors = np.zeros((len(aValues), len(bValues)))

        error_best = float("inf")

        for a in range(len(aValues)):
            for b in range(len(bValues)):
                p_y_x = self.estimate_p_y_x(p_y, self.summarizeByClass(Xtrain, Ytrain, aValues[a], bValues[b]), Xval)
                errors[a,b] = self.error_fun(p_y_x, Yval)
                if errors[a,b] < error_best:
                    error_best = errors[a,b]
                    bestA = aValues[a]
                    bestB = bValues[b]

        return error_best, bestA, bestB, errors

    def estimate_a_priori(self, yTrain):
        M = np.unique(yTrain).size
        N = len(yTrain)

        p_y = np.zeros((M,2), dtype='object')

        for idx, a in enumerate(np.unique(yTrain)):
            p_y[idx, 1] = a

        for a in range(M):
            for b in range(N):
                if yTrain[b] == p_y[a,1]:
                    p_y[a,0] = p_y[a,0] + 1
            p_y[a,0] = p_y[a,0]/N

        return p_y

    def separateByClass(self, features, classes):
        separated = {}
        for i in range(len(classes)):
            vector = features[i]
            if (classes[i] not in separated):
                separated[classes[i]] = []
            separated[classes[i]].append(vector)
        return separated

    def mean(self, numbers, a, b):
        return np.mean(numbers)+a+b

    def stdev(self, numbers, b):
        return math.fabs(np.std(numbers)+b)+0.0001

    def summarize(self, dataset, a, b):
        summaries = [(self.mean(attribute, a, b), self.stdev(attribute, b)) for attribute in zip(*dataset)]
        return summaries

    def summarizeByClass(self, features, classes, a, b):
        separated = self.separateByClass(features, classes)
        summaries = {}
        for classValue, instances in separated.items():
            summaries[classValue] = self.summarize(instances, a, b)
        return summaries

    def calculateProbability(self, x, mean, stdev):
        exponent = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(stdev, 2))))
        return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent

    def estimate_p_y_x(self, p_y, summaries, X):
        N = np.size(X, 0)
        M = len(p_y)

        p_y_x = np.zeros((N, M))

        iloczyn = 0

        for a in range(N):
            for b in range(M):
                for c in range(np.size(X, 1)):
                    mean, stdev = summaries[p_y[b,1]][c]
                    x = X[a,c]
                    try:
                        iloczyn += math.log(self.calculateProbability(x, mean, stdev))
                    except ValueError:
                        iloczyn += math.log(0.00001)
                iloczyn = iloczyn + math.log(p_y[b, 0])
                p_y_x[a, b] = iloczyn
                iloczyn = 0
        return p_y_x

    def error_fun(self, p_y_x, Y):
        error_val = 0

        for i in range(len(Y)):
            max = 0
            for j in range(np.size(p_y_x, 1)):
                if p_y_x[i,max] <= p_y_x[i,j]:
                    max = j
            if self.p_y[max,1] != Y[i]:
                error_val = error_val + 1

        error_val = error_val/len(Y)

        return error_val

    def error_values(self, Y, true_y):
        error_val = 0

        for i in range(len(true_y)):
            if Y[i] != true_y[i]:
                error_val = error_val + 1
        return error_val/len(true_y)

    def accuracy_values(self, Y, true_y):
        return 1 - self.error_values(Y, true_y)

    def accuracy(self, p_y_x, Y):
        return 1 - self.error_fun(p_y_x, Y)

    def predict(self, X):
        return self.estimate_p_y_x(self.p_y, self.p_x_y, X)

    def predictValues(self, X):
        predictedData = self.estimate_p_y_x(self.p_y, self.summaries, X)
        reasults = []
        for predict in predictedData:
            reasults.append(self.p_y[np.argmax(predict),1])
        return np.array(reasults)