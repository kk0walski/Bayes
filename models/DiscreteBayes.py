import numpy as np
from sklearn.model_selection import train_test_split


class DiscreteBayes:

    def fit(self, X, Y, bins, test_size=0.5):
        a_values = [1, 3, 5, 10, 20, 30, 50, 100, 150, 300, 500, 1000]
        b_values = [1, 3, 5, 10, 20, 30, 50, 100, 150, 300, 500, 1000]
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = test_size, random_state = 0)
        error_best, best_a, best_b, errors = self.modelSelection(X_train, X_test, y_train, y_test, bins, a_values, b_values)
        self.p_y = self.estimate_a_priori(y_train)
        self.p_x_y = self.estimate_p_x_y(X_train, y_train, bins, best_a, best_b);
        self.bins = bins

    def modelSelection(self, Xtrain, Xval, Ytrain, Yval, bins, aValues, bValues):
        errors = np.zeros((len(aValues), len(bValues)))

        error_best = float("inf")

        for a in range(len(aValues)):
            for b in range(len(bValues)):
                p_y_x = self.estimate_p_y_x(self.estimate_a_priori(Ytrain), self.estimate_p_x_y(Xtrain, Ytrain, bins, aValues[a], bValues[b]), bins, Xval)
                errors[a,b] = self.error_fun(p_y_x, Yval)
                if errors[a,b] < error_best:
                    error_best = errors[a,b]
                    bestA = aValues[a]
                    bestB = bValues[b]

        return error_best, bestA, bestB, errors

    def error_fun(self, p_y_x, Y):
        error_val = 0

        for i in range(len(Y)):
            max = 0
            for j in range(np.size(p_y_x, 1)):
                if p_y_x[i,max] <= p_y_x[i,j]:
                    max = j
            if max != Y[i]:
                error_val = error_val + 1

        error_val = error_val/len(Y)

        return error_val

    def accuracy(self, p_y_x, Y):
        return 1 - self.error_fun(p_y_x, Y)

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

    def estimate_p_x_y(self, Xtrain, Ytrain, bins, a, b):
        D = np.size(Xtrain, 1)
        M = np.unique(Ytrain).size

        p_x_y = np.zeros((M,D, len(bins)))

        sum1 = np.zeros(len(bins))
        sum2 = 0

        for i in range(M):
            for j in range(D):
                for k in range(len(Ytrain)):
                    for b in range(len(bins)):
                        if Ytrain[k] == i and Xtrain[k,j] == bins[b]:
                            sum1[b] = sum1[b] + 1
                        if Ytrain[k] == i:
                            sum2 = sum2 + 1
                sum1 = sum1 + a - 1
                sum2 = sum2 + a + b - 2
                p_x_y[i,j,:] = sum1/sum2
                sum1 = np.zeros(len(bins))
                sum2 = 0
        return p_x_y

    def estimate_p_y_x(self, p_y, p_x_1_y, bins, X):
        N = np.size(X,0)
        M = len(p_y)

        p_x_0_y = 1 - p_x_1_y
        p_y_x = np.zeros((N,M))

        iloczyn = 1
        sum = 0

        for a in range(N):
            for b in range(M):
                for c in range(np.size(X,1)):
                    for d in range(len(bins)):
                        if X[a,c] == bins[d]:
                            iloczyn = iloczyn*p_x_1_y[b,c,d]
                        else:
                            iloczyn = iloczyn*p_x_0_y[b,c,d]
                iloczyn = iloczyn*p_y[b]
                sum = sum + iloczyn
                iloczyn = 1
            for b in range(M):
                for c in range(np.size(X,1)):
                    for d in range(len(bins)):
                        if X[a,c] == bins[d]:
                            iloczyn = iloczyn*p_x_1_y[b,c,d]
                        else:
                            iloczyn = iloczyn*p_x_0_y[b,c,d]
                iloczyn = iloczyn*p_y[b]
                p_y_x[a,b] = float(iloczyn/float(sum))
                iloczyn = 1
            sum = 0
        return p_y_x

    def predict(self, X):
        return self.estimate_p_y_x(self.p_y, self.p_x_y, self.bins, X)