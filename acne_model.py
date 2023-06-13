# -*- coding: utf-8 -*-
"""
Created on Sat Jun 10 19:38:13 2023

@author: ankro
"""

# Acne detection
from tensorflow.keras.layers import Dense,Conv2D,Flatten,MaxPooling2D,Dropout# BatchNormalization,GlobalAvgPool2D
from tensorflow.keras import Sequential
from keras.preprocessing import image
import tensorflow as tf


train=image.ImageDataGenerator(rotation_range=45,
                        shear_range=0.1,
                        zoom_range=0.2,
                        horizontal_flip=True,
                        width_shift_range=0.1,
                        height_shift_range=0.1,
                        rescale = 1/255)

test=image.ImageDataGenerator(rotation_range=45,
                        shear_range=0.1,
                        zoom_range=0.2,
                        horizontal_flip=True,
                        width_shift_range=0.1,
                        height_shift_range=0.1,
                        rescale = 1/255)

train= train.flow_from_directory('C:/Users/ankro/Project/acne-data/Created Dataset/train/',
                                 class_mode='binary',
                                 batch_size=2,
                                target_size=(150, 150))
test= test.flow_from_directory('C:/Users/ankro/Project/acne-data/Created Dataset/test/',
                               class_mode='binary',
                               batch_size=2,
                              target_size=(150, 150))



model = Sequential()

model.add(Conv2D(filters = 4, kernel_size = 2,padding ='valid',strides=1, input_shape = (150, 150, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(filters = 4, kernel_size = 2,padding = 'valid',strides=1, activation='relu' ))
model.add(MaxPooling2D(pool_size=(2,2)))


model.add(Flatten())

model.add(Dense(32, activation = "relu"))
model.add(Dropout(0.5))

model.add(Dense(1, activation = "sigmoid"))



tf.keras.losses.BinaryCrossentropy(
                                from_logits=False,
                                label_smoothing=0.0,
                                axis=-1,
                                reduction="auto",
                                name="binary_crossentropy")


model.compile(optimizer='adam',loss='binary_crossentropy',metrics='accuracy')

filepath="weights-improvement-{epoch:02d}-{val_accuracy:.2f}.h5"
mc = tf.keras.callbacks.ModelCheckpoint(filepath, monitor="val_loss", mode="min", save_best_only=True, verbose=1)

model.fit_generator(train,epochs=20,validation_data=test,verbose=2,callbacks=[mc])


