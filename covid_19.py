# -*- coding: utf-8 -*-
"""COVID-19.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/14pBgwGqhhesWFPP5AYXPWUh21yLwoFpq
"""

!pip install torch
!pip install keras

import keras
from keras import layers
from keras.preprocessing import image
from keras.layers import Dense, GlobalAveragePooling2D
from keras.callbacks import CSVLogger
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from keras.applications.resnet import ResNet101
from keras.layers.convolutional import Conv2D, MaxPooling2D

INIT_LR = 1e-3
EPOCHS = 25
BS = 8

datagen = ImageDataGenerator(rescale = 1./255,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True)

train_data = datagen.flow_from_directory(
    directory = '/content/train',
    target_size = (224,224),
    class_mode='binary',
    shuffle=True
)

val_data = datagen.flow_from_directory(
    directory = '/content/test',
    target_size = (224,224),
    class_mode='binary',
    shuffle=True
)

val_data.class_indices

#!rm -rf `find -type d -name .ipynb_checkpoints`

model = Sequential([
                    # BLOCK 1
                    Conv2D(32, kernel_size=(3,3), activation='relu', input_shape = (224, 224, 3)),
                    Conv2D(64, kernel_size=(3,3), activation='relu'),
                    MaxPooling2D(pool_size=(2,2)),
                    Dropout(0.25),
                    # BLOCK 2
                    Conv2D(128, kernel_size=(3,3), activation='relu'),
                    MaxPooling2D(pool_size=(2,2)),
                    Dropout(0.25),
                    # BLOCK 3
                    Conv2D(128, kernel_size=(3,3), activation='relu'),
                    MaxPooling2D(pool_size=(2,2)),
                    Dropout(0.25),
                    # BLOCK 4
                    Conv2D(264, kernel_size=(3,3), activation='relu'),
                    MaxPooling2D(pool_size=(2,2)),
                    Dropout(0.25),
                    # block 5
                    Flatten(),
                    Dense(64, activation='relu'),
                    Dropout(0.5),
                    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.summary()

history = model.fit(
    train_data,
    epochs = 5
)

model.save('model.h5')

from keras.models import load_model

model = load_model('model.h5')

model.summary()

from PIL import Image

def predict(image_loc):
    sub = Image.open(image_loc)
    sub = sub.resize((224, 224))
    sub = np.array(sub)
    print(sub.shape)
    #sub = sub.reshape(-1, 224, 224, 3)
    sub = np.expand_dims(sub, axis=0)
    preds = model.predict(sub)
    #print(preds)
    return "normal" if preds else "covid" 
predict('/content/test/covid/covid21.jpeg')

print("hello")

