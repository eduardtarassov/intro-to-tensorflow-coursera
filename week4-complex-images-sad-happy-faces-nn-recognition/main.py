import tensorflow as tf
import os
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator


DESIRED_ACCURACY = 0.999


# # Root directory of the project
dirname = os.path.dirname(os.path.abspath(__file__))
#
# Directory with our training horse pictures
train_horse_dir = os.path.join(dirname, 'data/training/happy')

# Directory with our training human pictures
train_human_dir = os.path.join(dirname, 'data/training/sad')

# Directory with our validation horse pictures
# validation_horse_dir = os.path.join(dirname, 'data/validation/horses')

# Directory with our validation human pictures
# validation_human_dir = os.path.join(dirname, 'data/validation/humans')

train_horse_names = os.listdir(train_horse_dir)
print(train_horse_names[:10])

train_human_names = os.listdir(train_human_dir)
print(train_human_names[:10])

# validation_horse_names = os.listdir(validation_horse_dir)
# print(validation_horse_names[:10])
#
# validation_human_names = os.listdir(validation_human_dir)
# print(validation_human_names[:10])


class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if logs.get('acc') > 0.99:
      print("\nReached 99% accuracy so cancelling training!")
      self.model.stop_training = True


callbacks = myCallback()




# This Code Block should Define and Compile the Model
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
    # Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('sad') and 1 for the other ('happy')
    tf.keras.layers.Dense(1, activation='sigmoid')])



model.summary()

model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(lr=0.001),
              metrics=['acc'])


# This code block should create an instance of an ImageDataGenerator called train_datagen
# And a train_generator by calling train_datagen.flow_from_directory
# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(rescale=1 / 255)

train_dirname = dirname + '/data/training/'

# Flow training images in batches of 128 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
    train_dirname,  # This is the source directory for training images
    target_size=(300, 300),  # All images will be resized to 150x150
    batch_size=16,
    # Since we use binary_crossentropy loss, we need binary labels
    class_mode='binary')

# This code block should call model.fit_generator and train for
# a number of epochs.
history = model.fit_generator(
    train_generator,
    steps_per_epoch=8,
    epochs=15,
    verbose=1,
    callbacks=[callbacks])
# Expected output: "Reached 99.9% accuracy so cancelling training!""