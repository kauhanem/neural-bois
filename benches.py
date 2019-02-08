from progressBar import printProgressBar
import time

import numpy as np

from math import ceil

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


def benchmark(clf, n_splits, X, y, X_t=None, y_t=None, splitted=False, test_size=None):
    start = time.localtime()
    s_t = []
    
    for i in range(3,6):
        if start[i] < 10:
            a = "0" + str(start[i])
        else:
            a = str(start[i])
        s_t.append(a)

    print(f"Start time: {s_t[0]}:{s_t[1]}:{s_t[2]}")

    start = time.time()

    if splitted:
        channels = X[0].shape[1]
    
    else:
        channels = X.shape[1]
        cv = ShuffleSplit(n_splits=n_splits,
                          test_size=test_size, random_state=0)

    printProgressBar(0, channels+1,
                     prefix=f"Progressing channel 1/{channels+1}",
                     suffix="Complete", length=50)

    ch_scores = []

    for c in range(channels+1):
        printProgressBar(c+1, channels+1,
                            prefix=f"Progressing channel {c+1}/{channels+1}",
                            suffix="Complete", length=50)
        scores = []

        if splitted:
            for s in range(n_splits):
                if c != channels:
                    X_train = X[s][:,c,:]
                    X_test = X_t[s][:,c,:]
                else:
                    N, a, b = X[s].shape
                    X_train = X[s].reshape((N, a*b))

                    N, a, b = X_t[s].shape
                    X_test = X_t[s].reshape((N, a*b))
                
                y_train = y[s]
                y_test = y_t[s]

                clf.fit(X_train, y_train)
                score = clf.score(X_test, y_test)

                scores.append(score)

            scores = np.array(scores)
        
        else:
            if c != channels:
                X_channel = X[:,c,:]
            else:
                N,a,b = X.shape
                X_channel = X.reshape((N,a*b))
            
            scores = cross_val_score(clf, X_channel, y, cv=cv)

        ch_scores.append(scores)

    end = time.time()

    duration = end - start

    print(f"Duration: {duration:.3f}s")

    ch = 0
    for result in ch_scores:
        ch += 1
        
        if ch != channels+1:
            ch_str = f"Channel {ch}"
        else:
            ch_str = "All channels"

        mean = 100*result.mean()
        dev = 100*result.std()*2

        acc = f"{ch_str} accuracy: {mean:.3f} % (+/- {dev:.3f} %)"

        dev_bar = ceil(dev/5)*"○"
        bar = (round(mean.item()/5))*"■"
        bar = bar + (20-len(bar))*"□"
        
        print(f"{acc:50} {bar:25} {dev_bar}")
