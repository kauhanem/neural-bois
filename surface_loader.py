# Pattern Recognition and Machine Learning 2018-2
# Group: TUT_Group_7
# Authors: Jani Björklund, Matius Hurskainen, Mikko Kauhanen, Samu Lampinen

import glob
import os
from numpy import load, loadtxt, arange, array, asarray, zeros
from progressBar import printProgressBar

from squaternion import quat2euler

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, ShuffleSplit

def surface_loader(test_size=None,n_splits=5,folder='robotsurface'+os.sep,euler=False):
    X_train = load(folder+'X_train_kaggle.npy')
    X_test = load(folder+'X_test_kaggle.npy')
    y_train = loadtxt(folder+'y_train_final_kaggle.csv',
                      delimiter=',',usecols=(1),dtype='str')


    if euler:
        X_train_eulers = zeros((1703, 3, 128))
        for i in range(1703):
            for j in range(128):
                X_train_eulers[i, :, j] = quat2euler(
                    X_train[i, 4, j], X_train[i, 0, j], X_train[i, 1, j], X_train[i, 2, j], degrees=True)

        X_train = X_train_eulers

    le = LabelEncoder()
    le.fit(y_train)
    y_train = le.transform(y_train)

    data = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train
    }

    if test_size != None:
        X_train_s = []
        X_test_s = []
        y_train_s = []
        y_test_s = []

        groups = loadtxt(folder+'groups.csv',
                         delimiter=',',usecols=(1),dtype='int')
        
        rs = ShuffleSplit(n_splits=n_splits,
        test_size=test_size,random_state=0)
        
        X = arange(36)
        N = X_train.shape[0]
        
        split = 0

        for indices in rs.split(X):
            X_train_split = []
            y_train_split = []
            X_test_split = []
            y_test_split = []

            train_index = indices[0]
            split += 1

            print("Progressing split {}/{}".format(split,n_splits))

            printProgressBar(1,N,
            prefix="Progressing sample {}/{}".format(1,N),
            suffix='Complete',length=50)

            for i in range(N):
                printProgressBar(i,N-1,
                prefix="Progressing sample {}/{}".format(i+1,N),
                suffix='Complete',length=50)

                if groups[i] in train_index:
                    X_train_split.append(X_train[i,:,:])
                    y_train_split.append(y_train[i])
                else:
                    X_test_split.append(X_train[i,:,:])
                    y_test_split.append(y_train[i])
            
            X_train_s.append(asarray(X_train_split))
            y_train_s.append(asarray(y_train_split))
            X_test_s.append(asarray(X_test_split))
            y_test_s.append(asarray(y_test_split))

            print()

        X_train_s = array(X_train_s)
        y_train_s = array(y_train_s)
        X_test_s = array(X_test_s)
        y_test_s = array(y_test_s)
        
        data.update({
            'X_train_s': X_train_s,
            'X_test_s': X_test_s,
            'y_train_s': y_train_s,
            'y_test_s': y_test_s
        })
    return data,le
