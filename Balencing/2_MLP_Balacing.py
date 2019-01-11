import numpy as np
import keras
import csv
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, BatchNormalization
from keras.optimizers import Adam, SGD
from keras.callbacks import CSVLogger

img_size = 64
n_epochs = 500
batch_sizes = 1024
csv_name = 'mlp_training_ok.csv'

def train():
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
    model.add(Dense(800, activation='relu', input_shape=(img_size*img_size*3,)))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(1024, activation='relu'))
    model.add(BatchNormalization())    
    model.add(Dense(10, activation='softmax'))
    model.summary()

    model.compile(loss = 'categorical_crossentropy', optimizer = Adam(lr=0.000001), 
                  metrics=['accuracy'])  

    csv_logger = CSVLogger(csv_name)
    model.fit (train_data, train_target, epochs = n_epochs, 
                           batch_size = batch_sizes, validation_data=(test_data, test_target),
                           shuffle=True,
                           callbacks=[csv_logger])

    model.save('MLP_Model.h5')

    eva = model.evaluate(test_data,test_target,batch_sizes)
    print(eva)

if __name__ == '__main__':
    train()
