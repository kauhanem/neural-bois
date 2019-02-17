import time
import warnings
from math import floor

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import AdaBoostClassifier as ABC
from sklearn.ensemble import ExtraTreesClassifier as EFC
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.linear_model import LogisticRegression as LGR
from sklearn.linear_model import SGDClassifier as SGD
from sklearn.model_selection import GridSearchCV, cross_val_score, ShuffleSplit
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC

from benches import benchmark
from progressBar import printProgressBar
from surface_loader import surface_loader


def warn(*args, **kwargs):
    pass
warnings.warn = warn

# Kauhan branch

test_size = 0.2
n_splits = 2

print("- "*50)
data, le = surface_loader(test_size, n_splits, euler=True)
print("- "*50)

print("\nkeys:", len(data))

for i in data:
    print("{} shape: {}".format(i, data[i].shape))

    for j in data[i]:
        if np.array_equal(data[i].shape, np.array([n_splits, ])):
            print(j.shape)
    print()

print(np.unique(data['y_train']))
print(le.inverse_transform(np.unique(data['y_train'])))

X = X_train = data['X_train']
y = y_train = data['y_train']

X_test = data['X_test']

X_train_s = data['X_train_s']
y_train_s = data['y_train_s']

X_test_s = data['X_test_s']
y_test_s = data['y_test_s']

print(f"\nX: {X_train_s.shape}, y: {y_train_s.shape}, X_t: {X_test_s.shape}, y_t: {y_test_s.shape}")

# knn_parameters = {
#     "n_neighbors" : np.arange(3,7),
#     "weights" : ("uniform","distance")
# }
lda_parameters = {
    "solver" : ("svd","lsqr","eigen")
}

E = np.arange(-5,1,1)
C_range = [float(10**float(e)) for e in E]

svc_parameters = {
    "C" : C_range,
    "kernel" : ("linear","poly","rbf","sigmoid")
}
lgr_parameters = {
    "penalty" : ("l1","l2"),
    "C" : C_range
}
sgd_parameters = {
    "loss" : ("hinge", "log", "modified_huber", "squared_hinge", "perceptron", "squared_loss", "huber", "epsilon_insensitive", "squared_epsilon_insensitive"),
    "penalty" : ("none", "l2", "l1", "elasticnet")
}
rfc_parameters = {
    "n_estimators" : np.arange(50,201,10)
}
efc_parameters = {}
# abc_parameters = {}
# gbc_parameters = {}

classifiers = [
    [LDA(), "LDA", lda_parameters],
    [SVC(), "SVC", svc_parameters],
    [LGR(), "LogReg", lgr_parameters],
    [SGD(), "StochGradDesc", sgd_parameters],
    [RFC(), "Random Forest", rfc_parameters],
    [EFC(), "Extra Tree", efc_parameters]
]

# [KNN(), "KNearestNeighbor", knn_parameters],
# ,
#     [ABC(), "AdaBoost", abc_parameters],
#     [GBC(), "Gradient Boosting Classifier", gbc_parameters]

count = 0
clf_count = len(classifiers)
channels = data["X_train"].shape[1]

# T = Normalizer()

cv = ShuffleSplit(n_splits,test_size)

for clf,name,parameters in classifiers:
    count += 1

    print()
    print("- "*80)
    print(f"\nProgressing classifier {count}/{clf_count}: {name}\n")
    print("- "*80, "\n")

    N, a, b = X_train.shape
    X = X_train.reshape((N, a*b))
    # X_n = T.transform(X)
    
    classifier = GridSearchCV(clf,parameters,cv=5)

    score = cross_val_score(classifier,X,y,cv=cv)

    mean = 100*score.mean()
    dev = 200*score.std()
    
    print(f"CLF: {name}")
    print(f"{classifier}")
    print(f"Accuracy: {mean:.2f} % (+/- {dev:.2f} %)\n")

    # print("All data")
    # benchmark(clf, n_splits, X=X_train, y=y_train, test_size=test_size) # Whole data
    
    # print("\nSplitted data")
    # benchmark(clf, n_splits, X_train_s, y_train_s, X_test_s, y_test_s, splitted=True)  # Channel by channel
