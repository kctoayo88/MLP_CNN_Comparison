import numpy as np
import keras
import csv
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Conv2D, MaxPooling2D, Flatten, BatchNormalization
from keras.optimizers import RMSprop
from keras.callbacks import CSVLogger
from keras.preprocessing.image import ImageDataGenerator

img_size = 64
n_epochs = 5
batch_sizes = 128
n_steps_per_epoch = 1500
n_validation_steps = 1500
csv_logger = CSVLogger('cnn_training_Alex.csv')

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

#Build AlexNet model
model = Sequential()
 
#First Convolution and Pooling layer
model.add(Conv2D(96,(11,11),strides=(4,4),input_shape=(img_size, img_size, 3),padding='valid',activation='relu'))
model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))
    
#Second Convolution and Pooling layer
model.add(Conv2D(256,(5,5),strides=(1,1),padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))
    
#Three Convolution layer and Pooling Layer
model.add(Conv2D(384,(3,3),strides=(1,1),padding='same',activation='relu'))
model.add(Conv2D(384,(3,3),strides=(1,1),padding='same',activation='relu'))
model.add(Conv2D(256,(3,3),strides=(1,1),padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
    
#Fully connection layer
model.add(Flatten())
model.add(Dense(4096,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4096,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1000,activation='relu'))
model.add(Dropout(0.5))
    
#Classfication layer
model.add(Dense(10, activation='softmax'))

model.summary()

model.compile(loss="categorical_crossentropy", optimizer = RMSprop() ,
              metrics=["accuracy"])

model.fit_generator(train_generator,
                    epochs=n_epochs,
                    validation_data=test_generator,
                    steps_per_epoch = n_steps_per_epoch,
                    validation_steps = n_validation_steps,
                    callbacks=[csv_logger])

model.save('CNN_model_Alex.h5')