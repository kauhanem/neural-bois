import numpy as np
from squaternion import quat2euler
from scipy.integrate import cumtrapz

dtypes = {'soft_tiles': 0, 'hard_tiles': 1, 'carpet': 2, 'fine_concrete': 3,
              'soft_pvc': 4, 'tiled': 5, 'wood': 6, 'hard_tiles_large_space': 7,
              'concrete': 8}


def read_angle_data(X = [], Y = [], plot=True, DNN_format=True):
    # Load train data
    X_train = np.load('./robotsurface/X_train_kaggle.npy')
    
    # Convert quaternion angles to euler angles
    X_train_eulers = np.zeros((1703, 3, 128))
    for i in range(1703):
        for j in range(128):
            X_train_eulers[i, :, j] = quat2euler(X_train[i, 4, j], X_train[i, 0, j], X_train[i, 1, j], X_train[i, 2, j], degrees=True)
        
    # Load train labels
    y_train = np.loadtxt('./robotsurface/y_train_final_kaggle.csv', str, skiprows=1, delimiter=',')
    
    # Add data to lists
    for i in range(1703):
        x = X_train_eulers[i,:,:]
        X.append(x)
        if (DNN_format):
            # Create DNN-training suitable version of y: e.g. [0 0 0 1 0 0 0 0 0]
            y = np.zeros((9, 1))
            y = y.T
            n = dtypes[y_train[i, 1]]
            y[:, n] = 1
            Y.append(y)
        else:
            # Create sklearn-training suitable version of y: e.g. 'hard_tiles'
            Y.append(y_train[i, 1])
        
        
        
def read_all_data(X = [], Y = [], plot=True, DNN_format=True):
    # Load train data
    X_train = np.load('./robotsurface/X_train_kaggle.npy')
    
    # Convert quaternion angles to euler angles
    X_train_eulers = np.zeros((1703, 3, 128))
    for i in range(1703):
        for j in range(128):
            X_train_eulers[i, :, j] = quat2euler(X_train[i, 4, j], X_train[i, 0, j], X_train[i, 1, j], X_train[i, 2, j], degrees=True)
            
    
    # Convert linear accelerations to position
    X_train_pos = np.zeros((1703, 3, 128))
    X_train_vel = np.zeros((1703, 3, 128))
    t = np.linspace( 0, 1.28, 128)  # Guess of the time axis
    for i in range(1703):
        X_train_vel[i, 0, :] = cumtrapz(X_train[i, 7, :], t, initial=0)
        X_train_vel[i, 1, :] = cumtrapz(X_train[i, 8, :], t, initial=0)
        X_train_vel[i, 2, :] = cumtrapz(X_train[i, 9, :], t, initial=0)
        pos_x = cumtrapz(X_train_vel[i, 0, :], t, initial=0)
        pos_y = cumtrapz(X_train_vel[i, 1, :], t, initial=0)
        pos_z = cumtrapz(X_train_vel[i, 2, :], t, initial=0)
        X_train_pos[i, 0, :] = pos_x
        X_train_pos[i, 1, :] = pos_y
        X_train_pos[i, 2, :] = pos_z
        
        
    # Load train labels
    y_train = np.loadtxt('./robotsurface/y_train_final_kaggle.csv', str, skiprows=1, delimiter=',')
    
    # Add data to lists
    for i in range(1703):
        eulers = X_train_eulers[i,:,:]    # Euler angle data
        angvels = X_train[i, 4:7, :]      # Angular velocity data
        xy_pos = X_train_pos[i, :2, :]    # XY-position data
        xy_vel = X_train_vel[i, :2, :]    # XY-velocity data
        x = np.concatenate((eulers,angvels, xy_pos, xy_vel))
        X.append(x)
        if DNN_format:
            ## Create DNN-training suitable version of y: e.g. [0 0 0 1 0 0 0 0 0]
            y = np.zeros((9, 1))
            y = y.T
            n = dtypes[y_train[i, 1]]
            y[:, n] = 1
            Y.append(y)
        else:
            # Create sklearn-training suitable version of y: e.g. 'hard_tiles'
            Y.append(y_train[i, 1])
    print("Size")
    print(np.shape(X))
