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
from luku import read_angle_data, read_all_data, read_original_data
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
import warnings
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from squaternion import quat2euler
from scipy.integrate import cumtrapz



X_train = []
y_train = []
X_test = []
y_test = []

## TRAIN ON EULER ANGLE DATA ############################################################################################
#print("EULERS")
#read_angle_data(X, y, False, False)
#X = np.reshape(X, (1703, 384)) # roll + pitch + yaw
## Convert train data to a numpy array
#X = np.asarray(X)
#########################################################################################################################


## TRAIN ON EULER ANGLE AND XY-POSITION DATA ############################################################################
#print("EULERS + XY-POS")
#read_all_data(X_train, y_train, X_test, y_test, False, False) # roll + pitch + yaw + angvelX + angvelY + angvelZ + xpos + ypos + xvel + yvel
## Convert train data to a numpy array
#X_train = np.asarray(X_train)
#X_temp = []
#for i in range(np.shape(X_train)[0]):
#    a = X_train[i, 0, :]  # Roll
#    b = X_train[i, 1, :]  # Pitch
#    c = X_train[i, 2, :]  # Yaw
#    d = X_train[i, 6, :]  # Pos X
#    e = X_train[i, 7, :]  # Pos Y
#    x = np.concatenate((a, b, c, d, e))
#    X_temp.append(x)
## Convert train data to a numpy array
#X_train = np.asarray(X_temp)
#
## Convert test data to a numpy array
#X_test = np.asarray(X_test)
#X_temp = []
#for i in range(np.shape(X_test)[0]):
#    a = X_test[i, 0, :]  # Roll
#    b = X_test[i, 1, :]  # Pitch
#    c = X_test[i, 2, :]  # Yaw
#    d = X_test[i, 6, :]  # Pos X
#    e = X_test[i, 7, :]  # Pos Y
#    x = np.concatenate((a, b, c, d, e))
#    X_temp.append(x)
#    
## Convert train data to a numpy array
#X_test = np.asarray(X_temp)
#########################################################################################################################


## TRAIN ON VELOCITY NORMALIZED EULER ANGLE AND ANGULAR VELOCITY NORMALIZED EULER ANGLE DATA ############################
#read_all_data(X_train, y_train, X_test, y_test, False, False) # roll + pitch + yaw + angvelX + angvelY + angvelZ + xpos + ypos + xvel + yvel
## Convert train data to a numpy array
#X = np.asarray(X)
#
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
#########################################################################################################################


# TRAIN ON ORIGINAL DATA ############################################################################
print("ORIGINAL DATA")
read_original_data(X_train, y_train, X_test, y_test, False, False)
# Convert train data to a numpy array
X_train = np.asarray(X_train)
# Convert train data to a numpy array
X_test = np.asarray(X_test)

train_count = np.shape(X_train)[0]
test_count = np.shape(X_test)[0]

### RESHAPE #
## print("RESHAPE")
## Combine sensor data to a single vector
#X_train = np.reshape(X_train, (np.shape(X_train)[0], 1280)) # roll + pitch + yaw + x + y
#X_test = np.reshape(X_test, (np.shape(X_test)[0], 1280)) # roll + pitch + yaw + x + y
#
## MEANS #
# print("MEANS")
## Calculate means of each sensor data
#X_train_means = np.zeros((train_count, 10))
#for i in range(train_count):
#    for j in range(10):
#        X_train_means[i, j] = np.mean(X_train[i, j, :])
#    
#X_test_means = np.zeros((test_count, 10))
#for i in range(test_count):
#    for j in range(10):
#        X_test_means[i, j] = np.mean(X_test[i, j, :])
#        
#X_train = X_train_means
#X_test = X_test_means

#
# MEANS + SD
print("MEANS + SD")
# Calculate means and SDs of each sensor data
X_train_means = np.zeros((train_count, 10))
X_train_SDs = np.zeros((train_count, 10))
for i in range(train_count):
    for j in range(10):
        X_train_means[i, j] = np.mean(X_train[i, j, :])
        X_train_SDs[i, j] = np.std(X_train[i, j, :])
    
X_test_means = np.zeros((test_count, 10))
X_test_SDs = np.zeros((test_count, 10))
for i in range(test_count):
    for j in range(10):
        X_test_means[i, j] = np.mean(X_test[i, j, :])
        X_test_SDs[i, j] = np.std(X_test[i, j, :])
        
X_train = np.concatenate((X_train_means, X_train_SDs), axis=1)
X_test = np.concatenate((X_test_means, X_test_SDs), axis=1)
#######################################################################################################################


print("Train shape:")
print(np.shape(X_train))
print("Test shape:")
print(np.shape(X_test))
###########X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


warnings.filterwarnings("ignore")


# CLASSIFIERS
# KNN
print("KNN:")
model = KNeighborsClassifier(n_neighbors = 15, metric = "euclidean")
model.fit(X_train, y_train)
scores = cross_val_score(model, X_test, y_test, cv=5)
print(scores.mean())

# LDA
print("LDA:")
clf = LinearDiscriminantAnalysis()
clf.fit(X_train, y_train)
scores = cross_val_score(clf, X_test, y_test, cv=5)
print(scores.mean())

# SVM
print("SVM:")
svc = SVC(gamma='auto', kernel='linear', C=1)
svc.fit(X_train, y_train)
scores = cross_val_score(svc, X_test, y_test, cv=5)
print(scores.mean())

# LR
print("LR:")
model = LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=1000)
model.fit(X_train, y_train)
scores = cross_val_score(model, X_test, y_test, cv=5)
print(scores.mean())

## RF
print("RF:")
rf = RandomForestClassifier(n_estimators = 100)
rf.fit(X_train, y_train);
scores = cross_val_score(rf, X_test, y_test, cv=5)
print(scores.mean())

## ERF
print("ERF:")
clf = ExtraTreesClassifier(n_estimators=100)
clf.fit(X_train, y_train)
scores = cross_val_score(clf, X_test, y_test, cv=5)
print(scores.mean())

## ABC
print("ABC:")
clf = AdaBoostClassifier(n_estimators=100)
clf.fit(X_train, y_train)
scores = cross_val_score(clf, X_test, y_test, cv=5)
print(scores.mean())

# GBC
print("GBC:")
clf = GradientBoostingClassifier(n_estimators=100)
clf.fit(X_train, y_train)
scores = cross_val_score(clf, X_test, y_test, cv=5)
print(scores.mean())

# SVM_rbf
print("SVM_rbf:")
svc_rbf = SVC(gamma='auto', kernel='rbf', C=1)
svc_rbf.fit(X_train, y_train)
scores = cross_val_score(svc_rbf, X_test, y_test, cv=5)
print(scores.mean())



# TEST
#
## Load test data
#X_test = np.load('./robotsurface/X_test_kaggle.npy')
#
## Convert quaternion angles to euler angles
#X_test_eulers = np.zeros((1705, 3, 128))
#for i in range(1705):
#    for j in range(128):
#        X_test_eulers[i, :, j] = quat2euler(X_test[i, 4, j], X_test[i, 0, j], X_test[i, 1, j], X_test[i, 2, j], degrees=True)
#
## Convert linear accelerations to position
#X_load_pos = np.zeros((1705, 3, 128))
#X_load_vel = np.zeros((1705, 3, 128))
#t = np.linspace( 0, 1.28, 128)  # Guess of the time axis
#for i in range(1705):
#    X_load_vel[i, 0, :] = cumtrapz(X_test[i, 7, :], t, initial=0)
#    X_load_vel[i, 1, :] = cumtrapz(X_test[i, 8, :], t, initial=0)
#    X_load_vel[i, 2, :] = cumtrapz(X_test[i, 9, :], t, initial=0)
#    pos_x = cumtrapz(X_load_vel[i, 0, :], t, initial=0)
#    pos_y = cumtrapz(X_load_vel[i, 1, :], t, initial=0)
#    pos_z = cumtrapz(X_load_vel[i, 2, :], t, initial=0)
#    X_load_pos[i, 0, :] = pos_x
#    X_load_pos[i, 1, :] = pos_y
#    X_load_pos[i, 2, :] = pos_z
#        
## Reshape data
#X = []
#for i in range(1705):
#    x = X_test_eulers[i,:,:]    # Euler angle data
#    xx = X_test[i, 4:7, :]     # Angular velocity data
#    xxx = X_load_pos[i, :2, :]  # XY-position data
#    xxxx = X_load_vel[i, :2, :]    # XY-velocity data
#    xxxxx = np.concatenate((x, xx, xxx, xxxx))
#    X.append(xxxxx)
#X_test = np.reshape(X, (1705, (1280)))
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