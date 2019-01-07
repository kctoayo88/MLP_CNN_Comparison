import numpy as np
import keras
import csv
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Conv2D, MaxPooling2D, Flatten
from keras.models import Model
from keras.optimizers import RMSprop
from keras.callbacks import CSVLogger
from keras.preprocessing.image import ImageDataGenerator

img_size = 64
n_epochs = 10
batch_sizes = 4
n_steps_per_epoch = 1500
n_validation_steps = 1500
csv_logger = CSVLogger('cnn_training_cnn.csv')

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

model = Sequential()

model.add(Conv2D(filters = 16, kernel_size = (5, 5), 
                 padding = 'same', input_shape = (img_size, img_size, 3), 
                 activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Conv2D(filters = 36, kernel_size = (5, 5), 
                 padding = 'same', activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Flatten())
model.add(Dense(1024, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation = 'softmax'))

model.compile(optimizer = RMSprop(), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

model.summary()

model.fit_generator(train_generator,
                    epochs=n_epochs,
                    validation_data=test_generator,
                    steps_per_epoch = n_steps_per_epoch,
                    validation_steps = n_validation_steps,
                    callbacks=[csv_logger])

model.save('CNN_Model_cnn.h5')
