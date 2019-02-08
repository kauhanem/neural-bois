#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 11:46:01 2019

@author: matius
"""

# 3
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from luku import read_angle_data, read_all_data
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
import warnings
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from squaternion import quat2euler


X = []
y = []

## TRAIN ON EULER ANGLE DATA ############################################################################################
#read_angle_data(X, y, False, False)
#X = np.reshape(X, (1703, 384)) # roll + pitch + yaw
## Convert train data to a numpy array
#X = np.asarray(X)
#########################################################################################################################


# TRAIN ON EULER ANGLE AND XY-POSITION DATA ############################################################################
read_all_data(X, y, False, False) # roll + pitch + yaw + angvelX + angvelY + angvelZ + xpos + ypos + xvel + yvel
# Convert train data to a numpy array
X = np.asarray(X)
X_temp = []
for i in range(1703):
    a = X[i, 0, :]  # Roll
    b = X[i, 1, :]  # Pitch
    c = X[i, 2, :]  # Yaw
    d = X[i, 6, :]  # Pos X
    e = X[i, 7, :]  # Pos Y
    x = np.concatenate((a, b, c, d, e))
    X_temp.append(x)
    
# Convert train data to a numpy array
X = np.asarray(X_temp)
########################################################################################################################


# TRAIN ON VELOCITY NORMALIZED EULER ANGLE AND ANGULAR VELOCITY NORMALIZED EULER ANGLE DATA ############################
#read_all_data(X, y, False, False) # roll + pitch + yaw + angvelX + angvelY + angvelZ + xpos + ypos + xvel + yvel
# Convert train data to a numpy array
#X = np.asarray(X)

#X = X[:, :, 20:]  # Clip first 20 data points to reduce error caused by unknown initial linear velocities
#X_temp = []
#for i in range(1703):
#    a = (X[i, 1, :] / X[i, -2, :])    # Pitch / LinVel X
#    b = X[i, 0, :] / X[i, -1, :]     # Roll / LinVel Y
#    c = X[i, 3, :] / X[i, 0, :]    # AngVel X / Roll
#    d = X[i, 4, :] / X[i, 1, :]    # AngVel Y / Pitch
#    x = np.concatenate((a, b, c, d))
#    X_temp.append(x)
#
#X = np.asarray(X_temp)
########################################################################################################################


#X = np.reshape(X_temp, (1703, 432)) # roll + pitch + yaw + x + y

print(np.shape(X))
print(np.shape(y))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

## KNN
#model = KNeighborsClassifier(n_neighbors = 15, metric = "euclidean")
#model.fit(X_train, y_train)
#print("KNN: " + str(accuracy_score(y_test, model.predict(X_test))))
## LDA
#clf = LinearDiscriminantAnalysis()
#clf.fit(X_train, y_train)
#print("LDA: " + str(accuracy_score(y_test, clf.predict(X_test))))

## SVC
#svc = SVC(gamma='auto', kernel='linear', C=1)
#svc.fit(X_train, y_train)
#print("SVC: " + str(accuracy_score(y_test, svc.predict(X_test))))

## LR
#model = LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=1000)
#model.fit(X_train, y_train)
#print("LR: " + str(accuracy_score(y_test, model.predict(X_test))))

## RF
print("RF:")
rf = RandomForestClassifier(n_estimators = 100)
rf.fit(X_train, y_train);
#print("RF: " + str(accuracy_score(y_test, rf.predict(X_test))))
scores = cross_val_score(rf, X, y, cv=5)
print(scores.mean())

## ERF
print("ERF:")
clf = ExtraTreesClassifier(n_estimators=100)
clf.fit(X, y)
scores = cross_val_score(clf, X, y, cv=5)
print(scores.mean())

## ABC
#print("ABC:")
#clf = AdaBoostClassifier(n_estimators=100)
#clf.fit(X, y)
#scores = cross_val_score(clf, X, y, cv=5)
#print(scores.mean())

## GBC
#print("GBC:")
#clf = GradientBoostingClassifier(n_estimators=100)
#clf.fit(X, y)
#scores = cross_val_score(clf, X, y, cv=5)
#print(scores.mean())



# TEST

## Load test data
#X_test = np.load('./robotsurface/X_test_kaggle.npy')
#
## Convert quaternion angles to euler angles
#X_test_eulers = np.zeros((1705, 3, 128))
#for i in range(1705):
#    for j in range(128):
#        X_test_eulers[i, :, j] = quat2euler(X_test[i, 4, j], X_test[i, 0, j], X_test[i, 1, j], X_test[i, 2, j], degrees=True)
#        
#X_test = np.reshape(X_test_eulers, (1705, (384)))
#print("Test data shape:")
#print(np.shape(X_test))
#
## Submission file
#y_pred = rf.predict(X_test)
##labels = list(le.inverse_transform(y_pred))
#
#with open("submission.csv", "w") as fp:
#    fp.write("# Id,Surface\n")
#    for i, y_pred in enumerate(y_pred):
#        fp.write("%d,%s\n" % (i, y_pred))