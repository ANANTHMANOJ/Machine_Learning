# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 12:13:40 2019

@author: AN20027664
"""

#import Keras lib
from keras.model import Sequential       #for sequential modeling for CNN
from keras.layers import Convolution2D   # for convolution layers
from keras.layers import MaxPooling2D    # for maxpooling
from keras.layers import Flatten         #for flattening
from keras.layers import Dense           # for full-connections in hidden layer

classifier=Sequential()
classifier.add(Convolution2D(32,3,3,input_shape=(64,64,3),activation='relu')) #32,3,3 is number of filters having 3X3 size and Input_shape 64X64 for RGB(3)
classifier.add(MaxPooling2D(pool_size(2,2)))
classifier.add(Convolution2D(32,3,3,activation='relu')) #32,3,3 is number of filters having 3X3 size and Input_shape 64X64 for RGB(3)
classifier.add(MaxPooling2D(pool_size(2,2)))
classifier.add(Dense(output_dim=128,activation='relu'))
classifier.add(Dense(output_dim=1,activation='sigmoid'))

#compliling
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

from keras.preprosessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'data/validation',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(
        training_set,
        steps_per_epoch=8000,
        epochs=25,
        validation_data=test_set,
        validation_steps=2000)