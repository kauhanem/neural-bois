import numpy as np

from verkko import model_2D_definition, model_1D_definition
from luku import read_data
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD

if __name__ == '__main__':

    X = []
    Y = []

    read_data(X, Y)

    X = np.asarray(X)
    print(np.shape(X))
    X = np.reshape(X, (1703, 128, -1))
    Y = np.asarray(Y)
    Y = np.reshape(Y, (1703, 9))
    print(X.shape)

    # Load model
    #model = model_2D_definition()
    model = model_1D_definition()

    cp = ModelCheckpoint('/home/matius/Asiakirjat/Koulu/SGN/PR&ML/Harkkatyö/Verkko/Malli1--LinAccX--{epoch:02d}-{val_loss:.2f}.hdf5', save_best_only=True)

    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)


    # Train
    model.compile(loss='mean_squared_logarithmic_error', optimizer=sgd, metrics=['accuracy'])
    model.fit(X, Y, batch_size=60, epochs=10000, verbose=1, validation_split=0.2, callbacks=[cp], shuffle=True)
