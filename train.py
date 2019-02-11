import numpy as np

from verkko import model_2D_definition, model_1D_definition
from luku import read_angle_data, read_all_data, read_original_data
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD

data_dimensions = 10     # <------------ Change here the number of sensor channels

if __name__ == '__main__':
    
    X_train = []
    y_train = []
    X_test = []
    y_test = []

    #read_angle_data(X, Y, False)
    
#    # EULER + XY-POS -DATA ############################################################################################
#    read_all_data(X_train, y_train, X_test, y_test, False, True)
#    
#    # Select data
#    X_train = np.asarray(X_train)
#    X_temp = []
#    for i in range(np.shape(X_train)[0]):
#        a = X_train[i, 0, :]  # Roll
#        b = X_train[i, 1, :]  # Pitch
#        c = X_train[i, 2, :]  # Yaw
#        d = X_train[i, 6, :]  # Pos X
#        e = X_train[i, 7, :]  # Pos Y
#        x = np.concatenate((a, b, c, d, e))
#        X_temp.append(x)
#    # Convert train data to a numpy array
#    X_train = np.asarray(X_temp)
#    
#    # Convert test data to a numpy array
#    X_test = np.asarray(X_test)
#    X_temp = []
#    for i in range(np.shape(X_test)[0]):
#        a = X_test[i, 0, :]  # Roll
#        b = X_test[i, 1, :]  # Pitch
#        c = X_test[i, 2, :]  # Yaw
#        d = X_test[i, 6, :]  # Pos X
#        e = X_test[i, 7, :]  # Pos Y
#        x = np.concatenate((a, b, c, d, e))
#        X_temp.append(x)
#        
#     Convert train data to a numpy array
#    X_test = np.asarray(X_temp)
#    
#    train_count = np.shape(X_train)[0]
#    test_count = np.shape(X_test)[0]
#
#    X_train = np.reshape(X_train, (train_count, 128*data_dimensions, -1))     # Euler orientation data + (angular velocity data) + xy-position data
#    y_train = np.reshape(y_train, (train_count, 9))
#    X_test = np.reshape(X_test, (test_count, 128*data_dimensions, -1))     # Euler orientation data + (angular velocity data) + xy-position data
#    y_test = np.reshape(y_test, (test_count, 9))
#    ###################################################################################################################
    
    
    # TRAIN WITH ORIGINAL DATA #########################################################################################
    read_original_data(X_train, y_train, X_test, y_test, False, True)
    
    train_count = np.shape(X_train)[0]
    test_count = np.shape(X_test)[0]

    X_train = np.reshape(X_train, (train_count, 128*data_dimensions, -1))     # Euler orientation data + (angular velocity data) + xy-position data
    y_train = np.reshape(y_train, (train_count, 9))
    X_test = np.reshape(X_test, (test_count, 128*data_dimensions, -1))     # Euler orientation data + (angular velocity data) + xy-position data
    y_test = np.reshape(y_test, (test_count, 9))
    ####################################################################################################################
    

    # Load model
    #model = model_2D_definition()
    model = model_1D_definition()

    cp = ModelCheckpoint('/home/matius/Asiakirjat/Koulu/SGN/PR&ML/Harkkatyö/Verkko/Malli--Euler+PositionData--{epoch:02d}-{val_loss:.2f}-acc-{acc:.4f}-val_acc-{val_acc:.4f}.hdf5', save_best_only=True)

    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)


    # Train
    model.compile(loss='mean_squared_logarithmic_error', optimizer=sgd, metrics=['accuracy'])
    model.fit(X_train, y_train, batch_size=60, epochs=10000, verbose=1, validation_data=(X_test, y_test), callbacks=[cp], shuffle=True)
