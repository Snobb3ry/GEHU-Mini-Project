import imp
import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D
from keras.optimizers import adam_v2
from keras.layers import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator

train_dir = 'data/train'
val_dir = 'data/test'
train_datagen = ImageDataGenerator(rescale=1.5/255)
val_datagen = ImageDataGenerator(rescale=1.5/255)
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(48.48),
    batch_size=64,
    color_mode="greyscale",
    class_mode='categorical',
)
validation_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(48.48),
    batch_size=64,
    color_mode="greyscale",
    class_mode='categorical',
)

