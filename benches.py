from progressBar import printProgressBar
import time

import numpy as np

from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression as LR
from sklearn.linear_model import SGDClassifier as SGD

from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.ensemble import ExtraTreesClassifier as EFC
from sklearn.ensemble import AdaBoostClassifier as ABC
from sklearn.ensemble import GradientBoostingClassifier as GBC

from sklearn.model_selection import ShuffleSplit, cross_val_score


def benchmark(clf, n_splits, channels, X, y, X_t=None, y_t=None, test_size=None):
    start = time.time()

    if test_size == None:
        printProgressBar(0, channels,
                         prefix=f"Progressing channel 1/{channels}",
                         suffix="Complete", length=50)
        
        ch_scores = []

        for c in range(channels):
            printProgressBar(c+1, channels,
                                 prefix=f"Progressing channel {c+1}/{channels}",
                                 suffix="Complete", length=50)
            scores = []
            for s in range(n_splits):
                X_c = X[s][:, c, :]
                X_t_c = X_t[s][:, c, :]

                y_c = y[s]
                y_t_c = y_t[s]

                clf.fit(X_c, y_c)
                score = clf.score(X_t_c, y_t_c)

                scores.append(score)
            scores = np.array(scores)
            ch_scores.append(scores)
        
        end = time.time()

        duration = end - start
        
        print(f"Duration: {duration:.3f} s")

        ch = 0
        for result in ch_scores:
            ch += 1
            print(f"Channel {ch} accuracy: {100*result.mean():.3f} % (+/- {100*result.std()*2:.3f} %)")

    else:
        N, a, b = X.shape

        X = X.reshape((N,a*b))
        
        cv = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=0)
        ch_scores = cross_val_score(clf,X,y,cv=cv)

        end = time.time()

        duration = end - start

        print(f"Duration: {duration:.3f} s")
        print(f"Accuracy: {100*ch_scores.mean():.3f} % (+/- {100*ch_scores.std()*2:.3f} %)")