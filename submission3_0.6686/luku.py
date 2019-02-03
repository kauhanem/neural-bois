import numpy as np
import matplotlib.pyplot as plt
from squaternion import quat2euler


def read_data(X = [], Y = [], plot=True):
    dtypes = {'soft_tiles': 0, 'hard_tiles': 1, 'carpet': 2, 'fine_concrete': 3,
              'soft_pvc': 4, 'tiled': 5, 'wood': 6, 'hard_tiles_large_space': 7,
              'concrete': 8}
    
    # Load train data
    X_train = np.load('./robotsurface/X_train_kaggle.npy')
    
    # Convert quaternion angles to euler angles
    X_train_eulers = np.zeros((1703, 3, 128))
    for i in range(1703):
        for j in range(128):
            X_train_eulers[i, :, j] = quat2euler(X_train[i, 4, j], X_train[i, 0, j], X_train[i, 1, j], X_train[i, 2, j], degrees=True)
    
    # Plot angle data
    if plot:
        plt.figure()
        plt.ion()
        for i in range(1703):
            plt.subplot(331)
            plt.plot(X_train[i, 0, :])
            plt.subplot(332)
            plt.plot(X_train[i, 1, :])
            plt.subplot(334)
            plt.plot(X_train[i, 2, :])
            plt.subplot(335)
            plt.plot(X_train[i, 3, :])
            plt.subplot(337)
            plt.plot(X_train_eulers[i, 0, :])
            plt.subplot(338)
            plt.plot(X_train_eulers[i, 1, :])
            plt.subplot(339)
            plt.plot(X_train_eulers[i, 2, :])
            plt.show()
            plt.pause(5)
            plt.clf()
        
    # Load train labels
    y_train = np.loadtxt('./robotsurface/y_train_final_kaggle.csv', str, skiprows=1, delimiter=',')
    
    # Add data to lists
    for i in range(1703):
        x = X_train_eulers[i,:,:]
        X.append(x)
#        # Create DNN-training suitable version of y: e.g. [0 0 0 1 0 0 0 0 0]
        y = np.zeros((9, 1))
        y = y.T
        n = dtypes[y_train[i, 1]]
        y[:, n] = 1
        Y.append(y)
        # Create sklearn-training suitable version of y: e.g. 'hard_tiles'
#        Y.append(y_train[i, 1])

#read_data()