#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  3 00:26:05 2019

@author: matius
"""

import numpy as np
from squaternion import quat2euler
from keras.models import load_model


dtypes = {0: 'soft_tiles', 1: 'hard_tiles', 2: 'carpet', 3: 'fine_concrete',
          4: 'soft_pvc', 5: 'tiled', 6: 'wood', 7: 'hard_tiles_large_space',
          8: 'concrete'}

# Load model
model = load_model('Malli1--LinAccX--8275-0.03-acc_0.8216-val_acc_0.6686.hdf5')

# Load test data
X_test = np.load('./robotsurface/X_test_kaggle.npy')

# Convert quaternion angles to euler angles
X_test_eulers = np.zeros((1705, 3, 128))
for i in range(1705):
    for j in range(128):
        X_test_eulers[i, :, j] = quat2euler(X_test[i, 4, j], X_test[i, 0, j], X_test[i, 1, j], X_test[i, 2, j], degrees=True)
        
X_test = np.reshape(X_test_eulers, (1705, (384)))
print("Test data shape:")
print(np.shape(X_test))

# Submission file
y_pred = model.predict_classes(np.reshape(X_test, (1705, 384, -1)))
#labels = list(le.inverse_transform(y_pred))

with open("submission.csv", "w") as fp:
    fp.write("# Id,Surface\n")
    for i, y_pred in enumerate(y_pred):
        fp.write("%d,%s\n" % (i, dtypes[y_pred]))
