#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  3 00:26:05 2019

@author: matius
"""

import numpy as np
from squaternion import quat2euler
from keras.models import load_model
from scipy.integrate import cumtrapz

dtypes = {0: 'soft_tiles', 1: 'hard_tiles', 2: 'carpet', 3: 'fine_concrete',
          4: 'soft_pvc', 5: 'tiled', 6: 'wood', 7: 'hard_tiles_large_space',
          8: 'concrete'}


# Load model
model = load_model('Malli1--Euler+PositionData--4208-0.03-acc-0.8532-val_acc-0.7009.hdf5')

# Load test data
X_test = np.load('./robotsurface/X_test_kaggle.npy')

# Convert quaternion angles to euler angles
X_test_eulers = np.zeros((1705, 3, 128))
for i in range(1705):
    for j in range(128):
        X_test_eulers[i, :, j] = quat2euler(X_test[i, 4, j], X_test[i, 0, j], X_test[i, 1, j], X_test[i, 2, j], degrees=True)

# Convert linear accelerations to position
X_test_pos = np.zeros((1705, 3, 128))
t = np.linspace( 0, 12.8, 128)  # Guess of the time axis
for i in range(1705):
    vel_x = cumtrapz(X_test[i, 7, :], t, initial=0)
    vel_y = cumtrapz(X_test[i, 8, :], t, initial=0)
    vel_z = cumtrapz(X_test[i, 9, :], t, initial=0)
    pos_x = cumtrapz(vel_x, t, initial=0)
    pos_y = cumtrapz(vel_y, t, initial=0)
    pos_z = cumtrapz(vel_z, t, initial=0)
    X_test_pos[i, 0, :] = pos_x
    X_test_pos[i, 1, :] = pos_y
    X_test_pos[i, 2, :] = pos_z
        
# Reshape data
X = []
for i in range(1705):
    x = X_test_eulers[i,:,:]    # Euler angle data
    #xx = X_train[i, 4:7, :]      # Angular velocity data
    xxx = X_test_pos[i, :2, :]  # XY-position data
    xxxx = np.concatenate((x, xxx))
    X.append(xxxx)
X_test = np.reshape(X, (1705, (640)))
print("Test data shape:")
print(np.shape(X_test))

# Submission file
y_pred = model.predict_classes(np.reshape(X_test, (1705, 640, -1)))

with open("submission.csv", "w") as fp:
    fp.write("# Id,Surface\n")
    for i, y_pred in enumerate(y_pred):
        fp.write("%d,%s\n" % (i, dtypes[y_pred]))
