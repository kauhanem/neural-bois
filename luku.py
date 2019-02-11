import numpy as np
from squaternion import quat2euler
from scipy.integrate import cumtrapz
from sklearn.model_selection import GroupShuffleSplit

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
        
        
        
def read_all_data(X_train, y_train, X_test, y_test, plot=True, DNN_format=True):
    # Load train data
    X_load = np.load('./robotsurface/X_train_kaggle.npy')
    
    # Convert quaternion angles to euler angles
    X_load_eulers = np.zeros((1703, 3, 128))
    for i in range(1703):
        for j in range(128):
            X_load_eulers[i, :, j] = quat2euler(X_load[i, 4, j], X_load[i, 0, j], X_load[i, 1, j], X_load[i, 2, j], degrees=True)
            
    
    # Convert linear accelerations to position
    X_load_pos = np.zeros((1703, 3, 128))
    X_load_vel = np.zeros((1703, 3, 128))
    t = np.linspace( 0, 1.28, 128)  # Guess of the time axis
    for i in range(1703):
        X_load_vel[i, 0, :] = cumtrapz(X_load[i, 7, :], t, initial=0)
        X_load_vel[i, 1, :] = cumtrapz(X_load[i, 8, :], t, initial=0)
        X_load_vel[i, 2, :] = cumtrapz(X_load[i, 9, :], t, initial=0)
        pos_x = cumtrapz(X_load_vel[i, 0, :], t, initial=0)
        pos_y = cumtrapz(X_load_vel[i, 1, :], t, initial=0)
        pos_z = cumtrapz(X_load_vel[i, 2, :], t, initial=0)
        X_load_pos[i, 0, :] = pos_x
        X_load_pos[i, 1, :] = pos_y
        X_load_pos[i, 2, :] = pos_z
        
        
    # Load train labels
    y_load = np.loadtxt('./robotsurface/y_train_final_kaggle.csv', str, skiprows=1, delimiter=',')
    
    X = []
    Y = []
    
    # Add data to lists
    for i in range(1703):
        eulers = X_load_eulers[i,:,:]    # Euler angle data
        angvels = X_load[i, 4:7, :]      # Angular velocity data
        xy_pos = X_load_pos[i, :2, :]    # XY-position data
        xy_vel = X_load_vel[i, :2, :]    # XY-velocity data
        x = np.concatenate((eulers,angvels, xy_pos, xy_vel))
        X.append(x)
        if DNN_format:
            ## Create DNN-training suitable version of y: e.g. [0 0 0 1 0 0 0 0 0]
            y = np.zeros((9, 1))
            y = y.T
            n = dtypes[y_load[i, 1]]
            y[:, n] = 1
            Y.append(y)
        else:
            # Create sklearn-training suitable version of y: e.g. 'hard_tiles'
            Y.append(y_load[i, 1])
    
    # Load group data
    groups = np.loadtxt('./robotsurface/groups.csv', str, skiprows=1, delimiter=',')
    # Use only the  group info
    groups = groups[:, 1]
    print(np.shape(X))
    print(np.shape(Y))
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2)
    train_dataset, test_dataset = next(gss.split(X=X, y=Y, groups=groups))
    
    print("Train shape")
    print(np.shape(train_dataset))
    print("Test shape")
    print(np.shape(test_dataset))
    
    for i in range(np.shape(train_dataset)[0]):
        X_train.append(X[train_dataset[i]])
        y_train.append(Y[train_dataset[i]])
        
    for i in range(np.shape(test_dataset)[0]):
        X_test.append(X[test_dataset[i]])
        y_test.append(Y[test_dataset[i]])
        
    print("Train shape")
    print(np.shape(X_train))
    print("Test shape")
    print(np.shape(X_test))
    
    
    
def read_original_data(X_train, y_train, X_test, y_test, plot=True, DNN_format=True):
    # Load train data
    X_load = np.load('./robotsurface/X_train_kaggle.npy')
        
    # Load train labels
    y_load = np.loadtxt('./robotsurface/y_train_final_kaggle.csv', str, skiprows=1, delimiter=',')
    
    Y = []
    
    # Add data to lists
    for i in range(1703):
        if DNN_format:
            ## Create DNN-training suitable version of y: e.g. [0 0 0 1 0 0 0 0 0]
            y = np.zeros((9, 1))
            y = y.T
            n = dtypes[y_load[i, 1]]
            y[:, n] = 1
            Y.append(y)
        else:
            # Create sklearn-training suitable version of y: e.g. 'hard_tiles'
            Y.append(y_load[i, 1])
    
    # Load group data
    groups = np.loadtxt('./robotsurface/groups.csv', str, skiprows=1, delimiter=',')
    # Use only the  group info
    groups = groups[:, 1]
    print(np.shape(X_load))
    print(np.shape(Y))
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2)
    train_dataset, test_dataset = next(gss.split(X=X_load, y=Y, groups=groups))
    
    print("Train shape")
    print(np.shape(train_dataset))
    print("Test shape")
    print(np.shape(test_dataset))
    
    for i in range(np.shape(train_dataset)[0]):
        X_train.append(X_load[train_dataset[i]])
        y_train.append(Y[train_dataset[i]])
        
    for i in range(np.shape(test_dataset)[0]):
        X_test.append(X_load[test_dataset[i]])
        y_test.append(Y[test_dataset[i]])
        
    print("Train shape")
    print(np.shape(X_train))
    print("Test shape")
    print(np.shape(X_test))