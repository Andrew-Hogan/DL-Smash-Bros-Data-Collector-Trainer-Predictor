import csv
from random import shuffle
import cv2
from numpy import loadtxt
import numpy as np
import sklearn
import random
import os
import matplotlib.pyplot as plt

##define dict
control_letters = 'wasdxertfgvbijkluonm12345678'  ##13, 15 = j,l
control_array = list(control_letters)
values = dict()
for index, letter in enumerate(control_array):
   values[letter] = index

##import raw data
sampleimport = []
sampleimport = loadtxt('training_controls.txt', delimiter=',')
print (sampleimport.shape)

##reduce overpresent classes
counter = 0
appended = True
sampledrop_prob = 0 ##defines probability of dropping a left or right arrow
samples = np.zeros((0,28))
poss_sample = np.zeros((1,28))

for line in sampleimport:
   if sampleimport[counter, 12] == 1:
      if random.randint(1, 10) > sampledrop_prob:
         poss_sample[0] = sampleimport[counter]
         appended = True
      else:
         appended = False
   elif sampleimport[counter, 13] == 1:
      if random.randint(1, 10) > sampledrop_prob:
         poss_sample[0] = sampleimport[counter]
         appended = True
      else:
         appended = False
   elif sampleimport[counter, 15] == 1:
      if random.randint(1, 10) > sampledrop_prob:
         poss_sample[0] = sampleimport[counter]
         appended = True
      else:
         appended = False
   else:
      poss_sample[0] = sampleimport[counter]
      appended = True
   if appended == True:
      samples = np.append(samples, poss_sample, axis=0)
   counter += 1
print (samples.shape)
array_extra = np.zeros((len(samples),1))
samples = np.append(samples, array_extra, axis=1)

a = np.histogram(samples)
plt.hist(samples, bins='auto')
plt.title("test")
plt.show()

##add x value location to y to survive shuffle
counter_array = 0
while counter_array < (len(samples)):
    samples[counter_array, 28] = counter_array
    counter_array += 1
print (len(values))
print (samples.shape)

##shuffle
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
train_samples, validation_samples = train_test_split(samples, test_size=0.2)
print(len(train_samples))
print(len(validation_samples))

def preprocessgen(path):
   name = './training_data/training_' + str(int(path)) + '.jpg'
   image = np.array(cv2.cvtColor((cv2.imread(name)), cv2.COLOR_BGR2RGB))
   image = cv2.resize(image,(128, 128), interpolation=cv2.INTER_NEAREST)
   return image

def generator(samples, batch_size=128):
    num_samples = len(samples)
    global values
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            moves = []
            
            for batch_sample in batch_samples:
                #print (batch_sample[len(values)])
                image = preprocessgen(batch_sample[len(values)])
                images.append(image)
                move = np.delete(batch_sample, len(values), 0)
                moves.append(move)

            X_train = np.array(images)
            y_train = np.array(moves)
            yield sklearn.utils.shuffle(X_train, y_train)
            
# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=128)
validation_generator = generator(validation_samples, batch_size=128)

from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Cropping2D, Lambda, Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
import matplotlib.pyplot as plt


model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation 
model.add(Cropping2D(cropping=((0,0), (0,0)), input_shape = (128, 128, 3)))
model.add(Lambda(lambda x: (x/255) - 0.5))
model.add(Convolution2D(24, 5, 5,
                        border_mode='valid'))
model.add(Activation('relu'))
model.add(Convolution2D(36, 5, 5,
                       border_mode='valid'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Convolution2D(48, 3, 3,
                       border_mode='valid'))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3,
                       border_mode='valid'))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(1164))
model.add(Activation('relu'))
model.add(Dense(112))
model.add(Activation('relu'))
model.add(Dense(56))
model.add(Activation('relu'))
model.add(Dense(28))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')
history_object = model.fit_generator(train_generator, samples_per_epoch=
            (len(train_samples)), validation_data=validation_generator,
            nb_val_samples=(len(validation_samples)), nb_epoch=3, verbose=2)
model.save('model.h5')

print(history_object.history.keys())

#plot
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
