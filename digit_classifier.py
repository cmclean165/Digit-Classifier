# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import tensorflow as tf  
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, MaxPool2D, Dropout, Conv2D, Flatten
from tensorflow.keras.datasets import mnist

import numpy as np  

import matplotlib.pyplot as plt


# %%
# load the training data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train.shape


# %%
Layers = [
            Flatten(),
            Dense(254, activation='relu'),
            Dense(128, activation='relu'),
            Dense(10, activation='softmax')
        ]

model = Sequential(Layers)


# %%
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='accuracy')
model.fit(X_train, y_train, epochs=10)


# %%
location = np.random.randint(0, len(X_test)-1)

plt.imshow(X_test[location])
np.argmax(model.predict(X_test[location].reshape(-1, 28, 28)))

