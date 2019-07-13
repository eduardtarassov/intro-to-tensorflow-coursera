import tensorflow as tf
import matplotlib.pyplot as plt
print(tf.__version__)


class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('loss')<0.4):
      print("\nReached 60% accuracy so cancelling training!")
      self.model.stop_training = True

callbacks = myCallback()


# Loading MNIST fashion dataset from tf
mnist = tf.keras.datasets.fashion_mnist
# Splitting dataset into training and test images/labels
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

# Each image is 28x28, hence we have 60,000 labels
# And 28*28*60,000=47,040,000 items in images array
# plt.imshow(training_images[0])
# print(training_labels[0])
# print(training_images[0])
# print(training_labels.size)
# print(training_images.size)

# Normalizing values from 0-255 to 0-1
training_images  = training_images / 255.0
test_images = test_images / 255.0

# Defining the model
# "Sequential" defines a sequence of layers in NN
# Layer 1 = "Flatten" - turning images into 1d set
# Layer 2 = "Dense" - adds a layer of neurons
#   Each layer needs activation function to tell neurons what to do
#   "relu" = "if x>0 return x, else return 0"
#   (only passing values above 0 to next layer)
# Layer 3 = "softmax" takes a set of values and picks the biggest one:
#   [0.1, 0.1, 0.05, 0.1, 9.5, 0.1, 0.05, 0.05, 0.05] -> [0,0,0,0,1,0,0,0,0]
#   "Flattening" converts our 28*28 images into 784x1 array
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(128, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

# Building the model: compiling with optimizer and loss function
model.compile(optimizer = tf.train.AdamOptimizer(),
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])


# Training the model
#   The number of "epochs" mean how many times our dataset will iterate through NN
#   (too many epochs may lead to overfitting and this can affect "loss")
model.fit(training_images, training_labels, epochs=10, callbacks=[callbacks])


# Evaluating accuracy on unseen data
model.evaluate(test_images, test_labels)

# Prediction on test images. Classifications is a matrix where each row
# has 10 columns each label with probability
classifications = model.predict(test_images)
# print(classifications[2])
# print(test_labels[2])

