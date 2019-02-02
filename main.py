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

# Kauhan branch

test_size = 0.2
n_splits = 10

data,le = surface_loader(test_size,n_splits)

print("keys:",len(data))

for i in data:
    print("{} shape: {}".format(i,data[i].shape))

    for j in data[i]:
        if np.array_equal(data[i].shape,np.array([n_splits,])):
            print(j.shape)

print(np.unique(data['y_train']))
print(le.inverse_transform(np.unique(data['y_train'])))

X_train_s = data['X_train_s']
y_train_s = data['y_train_s']

X_test_s = data['X_test_s']
y_test_s = data['y_test_s']

# cv = ShuffleSplit(n_splits=6, test_size=test_size, random_state=0)

for s in range(n_splits):
    print("Progressing split {}/{}".format(s+1,n_splits))

    X = X_train_s[s]
    N,x,y = X.shape
    X = X.reshape((N,x*y))

    X_t = X_test_s[s]
    N_t,x,y = X_t.shape
    X_t = X_t.reshape((N_t,x*y))

    y = y_train_s[s]    
    y_t = y_test_s[s]

    scores = []

    printProgressBar(1,10,
    prefix="Progressing parameters {}/{}".format(1,10),
    suffix='Complete',length=50)

    for parameter in range(10):
        printProgressBar(parameter+1,10,
        prefix="Progressing parameters {}/{}".format(parameter+1,10),
        suffix='Complete',length=50)
        
        n_estimators = 25*parameter + 25
        clf = RFC(n_estimators=n_estimators)
        
        clf.fit(X,y)
        score = clf.score(X_t,y_t)
        scores.append(score)

    scores = np.array(scores)
    
    print("Accuracy: %0.2f (+/- %0.2f)\n" % (scores.mean(), scores.std() * 2))
