# -*- coding: utf-8 -*-
"""MNIST_hand_digit_recognition-Neural_Networks.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1_PNAkgdM_PQY7zfq6YXqvmBW9itmfMaU

# MNIST Hand Digit Recognition - using Neural Networks
"""

import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from sklearn.metrics import confusion_matrix

# %matplotlib inline

# Load MNIST hand digit recognition dataset
mnist = tf.keras.datasets.mnist

# Normalization of features
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train/255.0, x_test/255.0

# Training dataset information
print('Shape of training feature example: ', x_train[0].shape)
print('Shape of training examples: ', x_train.shape)
print('Number of training examples: ', len(x_train))
print('Shape of training labels: ', y_train.shape)
print('Number of training labels: ', len(y_train))

# Testing dataset information
print('Testing examples shape: ', x_test.shape)
print('Number of testing examples: ', len(x_test))
print('Testing labels shape: ', y_test.shape)
print('Number of testing lables: ', len(y_test))

# One hot encoding taining labels
one_hot_labels = tf.keras.utils.to_categorical(y_train, num_classes=10)

# Print one hot encoded labels
one_hot_labels

# Show first 4 hand digit recognition data
print('First 4 hand digit training examples:')
plt.subplot(221)
plt.imshow(x_train[0], cmap='gray')
plt.subplot(222)
plt.imshow(x_train[1], cmap='gray')
plt.subplot(223)
plt.imshow(x_train[2], cmap='gray')
plt.subplot(224)
plt.imshow(x_train[3], cmap='gray')
plt.show()

print('Corresponding output labels: ', y_train[0:4])

# Build keras sequential model
model = Sequential()
model.add(Dense(128, input_shape=(28, 28))) 
model.add(Flatten())
model.add(Activation('relu'))
model.add(Dense(64)) 
model.add(Activation('relu'))          
model.add(Dense(10, activation=tf.nn.softmax))

# model summary 
print(model.summary())

# Compile
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

"""# Fit the NN model without validation set"""

# fit the model without validation set
history_wo_val_set = model.fit(x_train, one_hot_labels, epochs=10, batch_size=500, verbose=1)

# Train test plot for model without validation set
def plot_history(history):
  '''
  Plots train validation set error and accuracy
  
  Arguments:
  history -- model history
  '''
  
  plt.subplot(211)
  plt.plot(history.epoch, np.array(history.history['loss']), label='Train Loss')
  plt.xlabel('Epoch')
  plt.ylabel('Error')
  plt.legend()

  plt.subplot(212)
  plt.plot(history.epoch, np.array(history.history['acc']), label='Train Acc')
  plt.xlabel('Epoch')
  plt.ylabel('Accuracy')
  plt.legend()
  
  plt.show()

# plot
plot_history(history_wo_val_set)

# Model predict labels
y_predicted = model.predict(x_test)

# Invert one-hot-encoded values
y_predicted = np.argmax(y_predicted, axis=1)
print(y_predicted)
print(y_test)

# Model predict labels
y_predicted = model.predict(x_test)

# Invert one-hot-encoded values
y_predicted = np.argmax(y_predicted, axis=1)
print(y_predicted)
print(y_test)

# Evaluate the model - loss value and metric (accuracy) value
one_hot_encode_test_label = tf.keras.utils.to_categorical(y_test, num_classes=10)
scores = model.evaluate(x_test, one_hot_encode_test_label, verbose=0)
print(scores)
print("Test Error: %.2f%%" % (scores[0]*100))
print("Test Accuracy: %.2f%%" % (scores[1]*100))

# Confusion matrix 
cmap = plt.cm.Blues
cm = confusion_matrix(y_test, invert)
plt.figure(figsize=(12,12))
sns.heatmap(cm, annot=True, annot_kws={"size": 12}, cmap='Blues')
plt.xlabel('True Lables', fontsize=20)
plt.ylabel('Predicted Labels', fontsize=20)
plt.show()

"""# Fit the NN model with validation set"""

# fit the model with validation set
history_w_val_set = model.fit(x_train, one_hot_labels, validation_split=0.2, epochs=10, batch_size=500, verbose=1)

# Train test plot for model with validation set
def plot_history(history):
  '''
  Plots train validation set error and accuracy
  
  Arguments:
  history -- model history
  '''
  
  plt.subplot(211)
  plt.plot(history.epoch, np.array(history.history['loss']), label='Train Loss')
  plt.plot(history.epoch, np.array(history.history['val_loss']), label='Val Loss')
  plt.xlabel('Epoch')
  plt.ylabel('Error')
  plt.legend()

  plt.subplot(212)
  plt.plot(history.epoch, np.array(history.history['acc']), label='Train Acc')
  plt.plot(history.epoch, np.array(history.history['val_acc']), label='Val Acc')
  plt.xlabel('Epoch')
  plt.ylabel('Accuracy')
  plt.legend()
  
  plt.show()

# plot
plot_history(history_w_val_set)

# Model predict labels
y_predicted = model.predict(x_test)

# Invert one-hot-encoded values
y_predicted = np.argmax(y_predicted, axis=1)
print(y_predicted)
print(y_test)

# Evaluate the model - loss value and metric (accuracy) value
one_hot_encode_test_label = tf.keras.utils.to_categorical(y_test, num_classes=10)
scores = model.evaluate(x_test, one_hot_encode_test_label, verbose=0)
print(scores)
print("Test Error: %.2f%%" % (scores[0]*100))
print("Test Accuracy: %.2f%%" % (scores[1]*100))

# Confusion matrix 
cmap = plt.cm.Blues
cm = confusion_matrix(y_test, invert)
plt.figure(figsize=(12,12))
sns.heatmap(cm, annot=True, annot_kws={"size": 12}, cmap='Blues')
plt.xlabel('True Labels', fontsize=20)
plt.ylabel('Predicted Labels', fontsize=20)
plt.show()

