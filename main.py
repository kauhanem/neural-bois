from surface_loader import surface_loader
from progressBar import printProgressBar
import numpy as np

from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression as LR

from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.ensemble import ExtraTreesClassifier as EFC
from sklearn.ensemble import AdaBoostClassifier as ABC
from sklearn.ensemble import GradientBoostingClassifier as GBC

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score, ShuffleSplit

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

# Kauhan branch

test_size = 0.2
n_splits = 5

data,le = surface_loader(test_size,n_splits)

# print("keys:",len(data))

# for i in data:
#     print("{} shape: {}".format(i,data[i].shape))

#     for j in data[i]:
#         if np.array_equal(data[i].shape,np.array([n_splits,])):
#             print(j.shape)

# print(np.unique(data['y_train']))
# print(le.inverse_transform(np.unique(data['y_train'])))

X_train_s = data['X_train_s']
y_train_s = data['y_train_s']

X_test_s = data['X_test_s']
y_test_s = data['y_test_s']

classifiers = [
    ["KNN", np.arange(1, 11).tolist()],
    ["LDA",[1]],
    ["SVC", [1]],
    ["LR", [1]],
    ["RandomForest", np.arange(50, 201, 50).tolist()],
    ["ExtraTree", np.arange(50, 201, 50).tolist()],
    ["AdaBoost", np.arange(50, 201, 50).tolist()],
    ["GradientBoost", np.arange(50, 201, 50).tolist()]
]

count = 0
luokittimia = len(classifiers)

for classifier in classifiers:
    count += 1
    name = classifier[0]
    parameters = classifier[1]

    print("- "*50)
    print("\nProgressing classifier {}/{}: {}\n".format(count,luokittimia,name))

    p_iter = 0
    for p in parameters:
        p_iter += 1
        print("Progressing parameter {}/{}".format(p_iter,len(parameters)))

        if name == "KNN":
            clf = KNN(n_neighbors=p)
        elif name == "LDA":
            clf = LDA()
        elif name == "SVC":
            clf = SVC()
        elif name == "LR":
            clf = LR()
        elif name == "RandomForest":
            clf = RFC(n_estimators=p)
        elif name == "ExtraTree":
            clf = EFC(n_estimators=p)
        elif name == "AdaBoost":
            clf = ABC(n_estimators=p)
        elif name == "GradientBoost":
            clf = GBC(n_estimators=p)

        printProgressBar(0, n_splits,
                         prefix="Progressed splits {}/{}".format(0, n_splits),
                         suffix='Complete', length=50)

        scores = []

        for s in range(n_splits):         
            X = X_train_s[s]
            N, x, y = X.shape
            X = X.reshape((N, x*y))

            X_t = X_test_s[s]
            N_t, x, y = X_t.shape
            X_t = X_t.reshape((N_t, x*y))

            y = y_train_s[s]
            y_t = y_test_s[s]
            
            clf.fit(X,y)
            score = clf.score(X_t,y_t)
            scores.append(score)

            printProgressBar(s+1, n_splits,
                             prefix="Progressed splits {}/{}".format(
                                 s+1, n_splits),
                             suffix='Complete', length=50)

        scores = np.array(scores)
        
        print("Accuracy: %0.2f (+/- %0.2f)\n" % (scores.mean(), scores.std() * 2))
