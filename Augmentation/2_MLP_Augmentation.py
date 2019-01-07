import numpy as np
import keras
import csv
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation 
from keras.optimizers import SGD
from keras.callbacks import CSVLogger
from keras.preprocessing.image import ImageDataGenerator

img_size = 64
n_epochs = 100
batch_sizes = 128
n_steps_per_epoch = 1500
n_validation_steps = 1500


def train(train_count):
    try:
        train_data = np.load('MLP_train_feature.npy') 
        train_target = np.load('MLP_train_target.npy') 
        test_data = np.load('MLP_test_feature.npy') 
        test_target = np.load('MLP_test_target.npy') 
    
        train_target = keras.utils.to_categorical(train_target,10)
        test_target = keras.utils.to_categorical(test_target,10)
        #print(train_target[0:10])

        '''
        print('train_data:')
        print(train_data)
        print('test_data:')
        print(test_data)
        print('train_target:')
        print(train_target)
        print('test_target:')
        print(test_target)
        '''

    except ValueError:
        print('Dataset files not founded ')



    model = Sequential() 
# Dense(64) is a fully-connected layer with 64 hidden units. 
# in the first layer, you must specify the expected input data shape: # here, 20-dimensional vectors. 
    model.add(Dense(512, activation='relu', input_shape=(img_size*img_size*3,)))   ## 全連結 64
    model.add(Dropout(0.1))   ### 為防止overtrained，只keep 50%之weights
    model.add(Dense(256, activation='relu')) ## 全連結 64
    model.add(Dense(256, activation='relu')) ## 全連結 64
    model.add(Dropout(0.1))
    model.add(Dense(256, activation='relu')) ## 全連結 64
    model.add(Dense(256, activation='relu')) ## 全連結 64
    model.add(Dropout(0.1))
    model.add(Dense(256, activation='relu')) ## 全連結 64
    model.add(Dense(10, activation='softmax')) ## 輸出 10 類
    model.summary()

    model.compile(loss = 'categorical_crossentropy', optimizer = 'adadelta', 
                  metrics=['accuracy'])  

    csv_logger = CSVLogger('mlp_training_test%s.csv' % (train_count))
    fit_score = model.fit (train_data, train_target, epochs = n_epochs, 
                           batch_size = batch_sizes, validation_data=(test_data, test_target),
                           callbacks=[csv_logger])

    model.save('MLP_model2.h5')

for train_count in range(0,4):
    train(train_count)