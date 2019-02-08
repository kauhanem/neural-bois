import warnings
def warn(*args, **kwargs):
    pass
warnings.warn = warn

from surface_loader import surface_loader
from progressBar import printProgressBar
from benches import benchmark

import time

from math import floor

import numpy as np

from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression as LGR
from sklearn.linear_model import SGDClassifier as SGD

from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.ensemble import ExtraTreesClassifier as EFC
from sklearn.ensemble import AdaBoostClassifier as ABC
from sklearn.ensemble import GradientBoostingClassifier as GBC

# Kauhan branch

test_size = 0.2
n_splits = 10

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

X_train = data['X_train']
y_train = data['y_train']

X_test = data['X_test']

X_train_s = data['X_train_s']
y_train_s = data['y_train_s']

X_test_s = data['X_test_s']
y_test_s = data['y_test_s']

print(f"\nX: {X_train_s.shape}, y: {y_train_s.shape}, X_t: {X_test_s.shape}, y_t: {y_test_s.shape}")

classifiers = [
    ["k Nearest Neighbors", np.arange(1, 11).tolist()],
    ["Linear Discriminant Analysis", [1]],
    ["Support Vector Classification", [1]],
    ["Logistic Regression", ['newton-cg','lbfgs','liblinear','sag','saga']],
    ["Stochastic Gradient Descent", ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron',
                                     'squared_loss', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive']],
    ["Random Forest", np.arange(50, 201, 50).tolist()],
    ["Extra Tree", np.arange(50, 201, 50).tolist()],
    ["AdaBoost", np.arange(50, 201, 50).tolist()],
    ["Gradient Boosting Classifier", np.arange(50, 201, 50).tolist()]
]

count = 0
clf_count = len(classifiers)
channels = data["X_train"].shape[1]

for classifier in classifiers:
    count += 1
    name = classifier[0]
    parameters = classifier[1]

    print()
    print("- "*80)
    print(f"\nProgressing classifier {count}/{clf_count}: {name}\n")
    print("- "*80, "\n")

    p_iter = 0
    for p in parameters:
        l = floor((100-len(str(p))-28)/4)
        p_iter += 1
        print()
        print(f"Progressing parameter {p_iter}/{len(parameters)}: {p}",
              "▬▬" * 2*l,
              "\n")

        if name == "k Nearest Neighbors":
            # print("S K I P P I N G")
            # continue
            clf = KNN(n_neighbors=p)

        elif name == "Linear Discriminant Analysis":
            # print("S K I P P I N G")
            # continue
            clf = LDA()

        elif name == "Support Vector Classification":
            # print("S K I P P I N G")
            # continue
            clf = SVC()

        elif name == "Logistic Regression":
            print("S K I P P I N G")
            continue
            clf = LGR()

        elif name == "Stochastic Gradient Descent":
            # print("S K I P P I N G")
            # continue
            clf = SGD(loss=p)

        elif name == "Random Forest":
            # print("S K I P P I N G")
            # continue
            clf = RFC(n_estimators=p)

        elif name == "Extra Tree":
            # print("S K I P P I N G")
            # continue
            clf = EFC(n_estimators=p)

        elif name == "AdaBoost":
            print("S K I P P I N G")
            continue
            clf = ABC(n_estimators=p)

        elif name == "Gradient Boosting Classifer":
            print("S K I P P I N G")
            break
            clf = GBC(n_estimators=p)

        print(clf)
        print()
        print("All data")
        benchmark(clf, n_splits, X=X_train, y=y_train, test_size=test_size) # Whole data
        
        print("\nSplitted data")
        benchmark(clf, n_splits, X_train_s, y_train_s, X_test_s, y_test_s, splitted=True)  # Channel by channel
