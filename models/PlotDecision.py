import numpy as np
import matplotlib.pyplot as plt
from itertools import product

class PlotDecision:
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def plot(self, classifier):
        # Plotting decision regions
        classifier.fit(self.X, self.Y)
        x_min, x_max = self.X[:, 0].min() - 1, self.X[:, 0].max() + 1
        y_min, y_max = self.X[:, 1].min() - 1, self.X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                             np.arange(y_min, y_max, 0.1))

        f, axarr = plt.subplots(1, 1, sharex='col', sharey='row', figsize=(10, 8))

        for idx in product([0,1], [0,1]):
            Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
            print(Z)
            Z = Z.reshape(xx.shape)
            axarr[idx[0], idx[1]].countourf(xx, yy, Z, alpha=0.4)
            axarr[idx[0], idx[1]].scatter(self,X[:, 0], self.X[:, 1], c=y,s=20, edgecolor='k')
            axarr[idx[0], idx[1]].set_title(tt)

        plt.show()