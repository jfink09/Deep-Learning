import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import itertools
import os
import shutil
import random
import glob
import matplotlib.pyplot as plt
import warnings

# Organize the data into training, valid, and test directories
'''os.chdir('dataset')
if os.path.isdir('dataset/normal') is False:
    os.makedirs('dataset/normal')
    os.makedirs('dataset/tumor')
    os.makedirs('valid/normal')
    os.makedirs('valid/tumor')
    os.makedirs('test/normal')
    os.makedirs('test/tumor')

    for c in random.sample(glob.glob('image*'), 90):
        shutil.move(c, 'dataset/normal')
    for c in random.sample(glob.glob('gg*'), 90):
        shutil.move(c, 'dataset/tumor')
    for c in random.sample(glob.glob('image*'), 70):
        shutil.move(c, 'valid/normal')
    for c in random.sample(glob.glob('gg*'), 70):
        shutil.move(c, 'valid/tumor')
    for c in random.sample(glob.glob('image*'), 50):
        shutil.move(c, 'test/normal')
    for c in random.sample(glob.glob('gg*'), 50):
        shutil.move(c, 'test/tumor')

os.chdir('../../')'''

train_path = 'dataset/dataset'
valid_path = 'dataset/valid'
test_path = 'dataset/test'

train_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
                .flow_from_directory(directory=train_path,target_size=(224,224),classes=['normal','tumor'],batch_size=10)
valid_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
                .flow_from_directory(directory=valid_path,target_size=(224,224),classes=['normal','tumor'],batch_size=10)
test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
                .flow_from_directory(directory=test_path,target_size=(224,224),classes=['normal','tumor'],batch_size=10,shuffle=False)

assert train_batches.n == 178
assert valid_batches.n == 138
assert test_batches.n == 98

imgs, labels = next(train_batches)

def plotImages(images_arr):
    fig, axes = plt.subplots(1,10,figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip(images_arr,axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

plotImages(imgs)
print(labels)