from keras import Sequential
from keras.layers.core import Dense, activations, Dropout
import warnings
import numpy as np
from keras.layers import Merge


def ignore_warn(*args ,**kwargs):
    pass


warnings.warn = ignore_warn
# 外层
model =Sequential()
model.add(Dense(1, input_dim=784, activation='softmax'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])


# 数据集
data = np.random.random((1000, 784))
labels = np.random.randint(2, size=(1000, 1))

# train the model,iterating on the data in batches
# 中间层
# of 32 samples
model.fit(data, labels, nb_epoch=10, batch_size=32)
left_branch = Sequential()
left_branch.add(Dense(32, input_dim=784))

right_branch = Sequential()
right_branch.add(Dense(32, input_dim=784))
merged = Merge([left_branch, right_branch], mode='concat')

model = Sequential()
model.add(merged)
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

