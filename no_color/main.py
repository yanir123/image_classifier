import tensorflow as tf
import numpy as np
from tensorflow import keras

## get premade dataset for uncolored images
fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels),
    (test_images, test_labels) = fashion_mnist.load_data()

## possible labels
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress',
    'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

## reduce the number of images to prevent overfitting
test_images = test_images / 255.0
train_images = train_images / 255.0

## create the model with keras,
## flatten the images, add relu (non linear) and softmax layers
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

## compile the model
model.compile(optimizer='adam',
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])

## train the model with the training dataset, using 5 epochs
model.fit(train_images, train_labels, epochs=5)

## evaluation to check the accuracy and the loss
test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print(test_accuracy)

## and finally prediction
prediction = model.predict(test_images)
print(np.argmax(prediction[0]))
print(test_labels[0])
