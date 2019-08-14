import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image, ImageFilter

import numpy as np
from tensorflow.keras.preprocessing import image
import glob

from os import listdir
from os.path import isfile, join

from filestack import Client

client = Client("APIKEY")

# # Root directory of the project
dirname = os.path.dirname(os.path.abspath(__file__))
#
# Directory with our training horse pictures
train_horse_dir = os.path.join(dirname, 'data/training/horse-or-human/horses')

# Directory with our training human pictures
train_human_dir = os.path.join(dirname, 'data/training/horse-or-human/humans')

# Directory with our validation horse pictures
validation_horse_dir = os.path.join(dirname, 'data/validation/horses')

# Directory with our validation human pictures
validation_human_dir = os.path.join(dirname, 'data/validation/humans')

train_horse_names = os.listdir(train_horse_dir)
print(train_horse_names[:10])

train_human_names = os.listdir(train_human_dir)
print(train_human_names[:10])

validation_horse_names = os.listdir(validation_horse_dir)
print(validation_horse_names[:10])

validation_human_names = os.listdir(validation_human_dir)
print(validation_human_names[:10])

#
# print('total training horse images:', len(os.listdir(train_horse_dir)))
# print('total training human images:', len(os.listdir(train_human_dir)))
# print('total validation horse images:', len(os.listdir(validation_horse_dir)))
# print('total validation human images:', len(os.listdir(validation_human_dir)))
#
#
# # Parameters for our graph; we'll output images in a 4x4 configuration
# nrows = 4
# ncols = 4
#
# # Index for iterating over images
# pic_index = 0
#
# # Set up matplotlib fig, and size it to fit 4x4 pics
# fig = plt.gcf()
# fig.set_size_inches(ncols * 4, nrows * 4)
#
# pic_index += 8
# next_horse_pix = [os.path.join(train_horse_dir, fname)
#                 for fname in train_horse_names[pic_index-8:pic_index]]
# next_human_pix = [os.path.join(train_human_dir, fname)
#                 for fname in train_human_names[pic_index-8:pic_index]]
#
# for i, img_path in enumerate(next_horse_pix+next_human_pix):
#   # Set up subplot; subplot indices start at 1
#   sp = plt.subplot(nrows, ncols, i + 1)
#   sp.axis('Off') # Don't show axes (or gridlines)
#
#   img = mpimg.imread(img_path)
#   plt.imshow(img)
#
# plt.show()


model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 300x300 with 3 bytes color
    # This is the first convolution
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(300, 300, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The third convolution
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The fourth convolution
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The fifth convolution
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'),
    # Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('horses') and 1 for the other ('humans')
    tf.keras.layers.Dense(1, activation='sigmoid')])

model.summary()

model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(lr=0.001),
              metrics=['acc'])

train_dirname = dirname + '/data/training/horse-or-human/'
validation_dirname = dirname + '/data/validation/'

# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(rescale=1 / 255)
validation_datagen = ImageDataGenerator(rescale=1 / 255)

# Flow training images in batches of 128 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
    train_dirname,  # This is the source directory for training images
    target_size=(300, 300),  # All images will be resized to 150x150
    batch_size=16,
    # Since we use binary_crossentropy loss, we need binary labels
    class_mode='binary')

# Flow training images in batches of 128 using train_datagen generator
validation_generator = validation_datagen.flow_from_directory(
    validation_dirname,  # This is the source directory for training images
    target_size=(300, 300),  # All images will be resized to 150x150
    batch_size=32,
    # Since we use binary_crossentropy loss, we need binary labels
    class_mode='binary')

history = model.fit_generator(
    train_generator,
    steps_per_epoch=8,
    epochs=15,
    verbose=1,
    validation_data=validation_generator,
    validation_steps=8)

upload_dirname = dirname + '/data/training/horse-or-human-test/'
upload_path = 'C:/Users/Eduard/PycharmProjects/intro-to-tensorflow-coursera/week4-complex-images-horse-human-nn-recognition/data/training/horse-or-human-test/'

# list filenames in dir
onlyfiles = [f for f in listdir(upload_path) if isfile(join(upload_path, f))]

print(onlyfiles)

# Iterating through all test images and making preditions
for fn in onlyfiles:
    # predicting images
    img = image.load_img(upload_path + fn, target_size=(300, 300))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    images = np.vstack([x])
    classes = model.predict(images, batch_size=10)
    print(classes[0])
    if classes[0] > 0.5:
        print(fn + " is a human")
    else:
        print(fn + " is a horse")
