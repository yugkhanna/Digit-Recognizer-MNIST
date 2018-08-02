import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pandas as pd

from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
from keras.callbacks import LearningRateScheduler

train_data = pd.read_csv("data/train.csv")
test_data = pd.read_csv("data/test.csv")

Y_train = train_data["label"]
X_train = train_data.drop(labels = ["label"],axis = 1)
X_train = X_train / 255.0
X_test = test_data / 255.0
X_train = X_train.values.reshape(-1,28,28,1)
X_test = X_test.values.reshape(-1,28,28,1)
Y_train = to_categorical(Y_train, num_classes = 10)

datagen = ImageDataGenerator(
        rotation_range=10,
        zoom_range = 0.1,
        width_shift_range=0.1,
        height_shift_range=0.1)

X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.2)

model = Sequential()

model.add(Conv2D(32, kernel_size = 3, activation='relu', input_shape = (28, 28, 1)))
model.add(BatchNormalization())
model.add(Conv2D(32, kernel_size = 3, activation='relu'))
model.add(MaxPool2D(2,2))
model.add(Dropout(0.4))

model.add(Conv2D(64, kernel_size = 3, activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size = 5, strides=2, padding='same', activation='relu'))
model.add(Dropout(0.4))

model.add(Conv2D(128, kernel_size = 4, activation='relu'))
model.add(BatchNormalization())

model.add(Flatten())
model.add(Dropout(0.4))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer = Adam(lr=0.001), metrics=["accuracy"])

annealer = LearningRateScheduler(lambda x: 1e-3 * 0.95 ** x)
model_try = model.fit_generator(datagen.flow(X_train, Y_train, batch_size=32), steps_per_epoch=500, epochs=20, verbose=1, validation_data=(X_val, Y_val), callbacks=[annealer])

predictions = model.predict(X_test)
predictions = np.argmax(predictions,axis = 1)
predictions = pd.Series(predictions, name="Label")
submit = pd.concat([pd.Series(range(1,28001),name = "ImageId"),predictions],axis = 1)
submit.to_csv("result.csv",index=False)
