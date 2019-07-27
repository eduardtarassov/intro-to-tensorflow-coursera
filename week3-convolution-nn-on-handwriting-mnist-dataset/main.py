import tensorflow as tf


# Original loss is 0.0277 and accuracy is 0.9916 on training data
# Original loss is 0.0712 and accuracy is 0.9776 on test data
# Convolutional NN loss is 0.019 and accuracy is 0.994 on training data
# Convolutional NN loss is 0.0331 and accuracy is 0.9883 on test data


# Need this in order to fix the CUDNN_STATUS_INTERNAL_ERROR
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.InteractiveSession(config=config)

# Loading Handwriting mnist
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_test)
print("------------------------------------------")



x_train = x_train.reshape(60000, 28, 28, 1)
x_train = x_train / 255.0
x_test = x_test.reshape(10000, 28, 28, 1)
x_test = x_test/255.0



#Added convolution + pooling + convolution + pooling = increased accuracy significantly.
model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()


# Too many epochs might cause overfitting - network learns the data from the training set really well,
# but it's too specialised to only that data, and as a result is less effective at seeing other data
model.fit(x_train, y_train, epochs=10)
print("-----------------------------------------------------")
model.evaluate(x_test, y_test)

print("=====================================================")
model.predict(x_test)


import matplotlib.pyplot as plt
f, axarr = plt.subplots(3,2)
FIRST_IMAGE=0
SECOND_IMAGE=7
THIRD_IMAGE=26
CONVOLUTION_NUMBER = 2
layer_outputs = [layer.output for layer in model.layers]
activation_model = tf.keras.models.Model(inputs = model.input, outputs = layer_outputs)
for x in range(0,2):
  f1 = activation_model.predict(x_test[FIRST_IMAGE].reshape(1, 28, 28, 1))[x]
  axarr[0,x].imshow(f1[0, : , :, CONVOLUTION_NUMBER], cmap='inferno')
  axarr[0,x].grid(False)
  f2 = activation_model.predict(x_test[SECOND_IMAGE].reshape(1, 28, 28, 1))[x]
  axarr[1,x].imshow(f2[0, : , :, CONVOLUTION_NUMBER], cmap='inferno')
  axarr[1,x].grid(False)
  f3 = activation_model.predict(x_test[THIRD_IMAGE].reshape(1, 28, 28, 1))[x]
  axarr[2,x].imshow(f3[0, : , :, CONVOLUTION_NUMBER], cmap='inferno')
  axarr[2,x].grid(False)
plt.show()
