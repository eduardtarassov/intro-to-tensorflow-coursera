import tensorflow as tf
print(tf.__version__)
mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

# Reshaping 60,000 28x28x1 items in a list into a 4D list that is 60,000x28x28x1.
# This is because convolution requires single tensor.
training_images=training_images.reshape(60000, 28, 28, 1)
print(training_images)
training_images=training_images / 255.0
test_images = test_images.reshape(10000, 28, 28, 1)
test_images=test_images/255.0

#Added convolution + pooling + convolution + pooling = increased accuracy significantly.
model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(28, 28, 1)),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2,2),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()
# Too many epochs might cause overfitting - network learns the data from the training set really well,
# but it's too specialised to only that data, and as a result is less effective at seeing other data
model.fit(training_images, training_labels, epochs=5)
test_loss = model.evaluate(test_images, test_labels)