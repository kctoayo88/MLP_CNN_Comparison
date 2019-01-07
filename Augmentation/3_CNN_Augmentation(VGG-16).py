import numpy as np
import keras
import csv
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Conv2D, MaxPooling2D, Flatten
from keras.optimizers import RMSprop
from keras.callbacks import CSVLogger
from keras.preprocessing.image import ImageDataGenerator

img_size = 64
n_epochs = 5
batch_sizes = 64
n_steps_per_epoch = 1500
n_validation_steps = 1500
#csv_logger = CSVLogger('cnn_training_vgg_test.csv')
model_file_name = 'CNN_Model_vgg_test.h5'

def train(train_count):
    try:
        train_data = np.load('CNN_train_feature.npy') 
        train_target = np.load('CNN_train_target.npy') 
        test_data = np.load('CNN_test_feature.npy') 
        test_target = np.load('CNN_test_target.npy') 
    
        #print(train_target.shape)
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


    train_datagen = ImageDataGenerator()
    train_generator = train_datagen.flow(train_data, train_target, batch_size = batch_sizes)
    test_datagen = ImageDataGenerator()
    test_generator = test_datagen.flow(test_data, test_target, batch_size = batch_sizes)

    ## Build VGGNet
    model = Sequential()

    model.add(Conv2D(32, (3, 3), padding='same',input_shape=(img_size, img_size, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    model.add(Dense(10, activation='softmax'))

    model.summary()

    model.compile(loss="categorical_crossentropy", optimizer = RMSprop() ,
                  metrics=["accuracy"])


    csv_logger = CSVLogger('cnn_training_vgg_test%s.csv' % (train_count))
    model.fit_generator(train_generator,
                        epochs=n_epochs,
                        validation_data=test_generator,
                        steps_per_epoch = n_steps_per_epoch,
                        validation_steps = n_validation_steps,
                        callbacks=[csv_logger])


    model.save(model_file_name)

for train_count in range(0,4):
    train(train_count)