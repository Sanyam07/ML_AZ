import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
from util import Util
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.datasets import cifar10, mnist, fashion_mnist

# get_data
# fashion_mnist = keras.datasets.fashion_mnist
# mnist = keras.datasets.mnist
# cifar10 = keras.datasets.cifar10

X_train, X_valid, X_test, y_train, y_valid, y_test = Util.prep_keras_data(mnist)
X_train = X_train.reshape(-1,28,28,1)
X_valid = X_valid.reshape(-1,28,28,1)
X_test = X_test.reshape(-1,28,28,1)


# # build base model
# model = keras.models.Sequential()
# model.add(keras.layers.Flatten(input_shape=[28,28]))
# model.add(keras.layers.Dense(300, activation='relu'))
# model.add(keras.layers.Dense(200, activation='relu'))
# model.add(keras.layers.Dense(10, activation='softmax'))

# input_shape = X_train.shape[1]*X_train.shape[2]
#
# input_ = keras.layers.Input(shape=[28,28])
# flat = keras.layers.Flatten(input_shape=[28,28])(input_)
# hidden1 = keras.layers.Dense(30, activation='relu')(flat)
# hidden2 = keras.layers.Dense(30, activation='relu')(hidden1)
# concat = keras.layers.Concatenate()([flat, hidden2])
# output = keras.layers.Dense(10, activation='softmax')(concat)
# model = keras.Model(inputs=[input_], outputs=[output])

###################################
# model = keras.models.Sequential()
# model.add(Conv2D(6,
#                  kernel_size=(5,5),
#                  strides=1,
#                  padding='same',
#                  input_shape=(28,28,1)))
# model.add(Activation('tanh'))
# model.add(AveragePooling2D(pool_size=(2,2),
#                            strides=2))
# model.add(Activation('tanh'))
# model.add(Conv2D(16,
#                  kernel_size=(5,5),
#                  strides=1))
# model.add(Activation('tanh'))
# model.add(AveragePooling2D(pool_size=(2,2),
#                            strides=2))
# model.add(Activation('tanh'))
# model.add(Conv2D(120,
#                  kernel_size=(5,5),
#                  strides=1))
# model.add(Activation('tanh'))
# model.add(Dense(84))
# model.add(Activation('tanh'))
# model.add(Dense(10, activation='softmax'))
# model.add(Activation('softmax'))

###############
model = keras.models.Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=X_train.shape[1:]))
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
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Activation('softmax'))


model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

run_logdir = Util.get_run_logdir()
tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)

history = model.fit(X_train, y_train,
                    epochs=30,
                    validation_data=(X_valid, y_valid),
                    callbacks=[tensorboard_cb])
pd.DataFrame(history.history).plot()
model.evaluate(X_test, y_test)

