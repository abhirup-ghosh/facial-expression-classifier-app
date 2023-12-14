#!/usr/bin/env python
# coding: utf-8

# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, log_loss, accuracy_score
import tensorflow as tf
from tensorflow import keras
import pickle

import warnings
warnings.filterwarnings("ignore")

# Load dataset and clean column names
print('... loading data')
data = pd.read_csv('../data/challenges-in-representation-learning-facial-expression-recognition-challenge/icml_face_data.csv')
data.columns = data.columns.str.strip()

# Define a function to prepare data for modeling
def prepare_data(data):
    """ 
    Prepare data for modeling
    input: data frame with labels and pixel data
    output: image and label array 
    """
    image_array = np.zeros(shape=(len(data), 48, 48))
    image_label = np.array(list(map(int, data['emotion'])))

    for i, row in enumerate(data.index):
        try:
            image = np.fromstring(data.loc[row, 'pixels'], dtype=int, sep=' ')
            image = np.reshape(image, (48, 48))
            image_array[i] = image
        except ValueError:
            continue

    return image_array, image_label

# Define input shape and emotion/class labels
print('... initialising input data')
input_shape = (48,48,1)
class_labels = ['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sadness', "Surprise"]
class_dict = dict(zip(range(7), class_labels))
num_classes = len(class_labels)

# Prepare training, validation, and test datasets
print('... preparing data')
train_image_array, train_image_label = prepare_data(data[data['Usage']=='Training'])
val_image_array, val_image_label = prepare_data(data[data['Usage']=='PublicTest'])
test_image_array, test_image_label = prepare_data(data[data['Usage']=='PrivateTest'])

# Rescale image arrays
print('... rescaling data')
train_image_array = train_image_array/255.
val_image_array = val_image_array/255.
test_image_array = test_image_array/255.

# One-hot encode target classes
print('... encoding target')
y_train_one_hot = tf.keras.utils.to_categorical(train_image_label, num_classes=num_classes)
y_val_one_hot = tf.keras.utils.to_categorical(val_image_label, num_classes=num_classes)
y_test_one_hot = tf.keras.utils.to_categorical(test_image_label, num_classes=num_classes)

# Define a function to print evaluation metrics
def print_metrics(labels, predictions):
    """
    Prints evaluation metrics.

    Parameters:
    - labels: True labels
    - predictions: Predicted labels

    Returns:
    - str: Formatted string containing evaluation metrics
    """
    predictions_labels = np.argmax(predictions, axis=1)
    return (f"LogLoss: {log_loss(labels, predictions):0.2f}| Accuracy: {accuracy_score(labels, predictions_labels):0.2f} | Precision: {precision_score(labels, predictions_labels, average='macro'):0.2f} | Recall: {recall_score(labels, predictions_labels, average='macro'):0.2f}")

# Define a function to create and compile a CNN model
def make_model(input_shape=(48, 48, 1),
               num_classes=7,
               metrics=['accuracy'],
               learning_rate=1e-4):
    """
    Create and compile a CNN model for image classification.

    Parameters:
    - input_shape: Tuple, shape of input images.
    - num_classes: Integer, number of output classes.
    - metrics: List of metrics to monitor during training.
    - learning_rate: Float, learning rate for the optimizer.

    Returns:
    - Keras model: Compiled CNN model.
    """

    # Initialize the CNN
    model = keras.models.Sequential()

    # Data augmentation layers
    # model.add(keras.layers.RandomFlip("horizontal", input_shape=input_shape))
    # model.add(keras.layers.RandomTranslation(height_factor=0.1, width_factor=0.1))
    # model.add(keras.layers.RandomContrast(factor=0.1))

    # Convolutional layers
    model.add(keras.layers.Conv2D(64,(3,3), padding='same', input_shape=input_shape))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Dropout(0.25))

    model.add(keras.layers.Conv2D(128,(5,5), padding='same'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Dropout(0.25))

    model.add(keras.layers.Conv2D(512,(3,3), padding='same'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Dropout(0.25))

    model.add(keras.layers.Conv2D(512,(3,3), padding='same'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Dropout(0.25))

    # Flatten the output
    model.add(keras.layers.Flatten())

    # Fully connected layers
    model.add(keras.layers.Dense(256))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Dropout(0.25))

    model.add(keras.layers.Dense(512))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Dropout(0.25))

    # Output layer with softmax activation
    model.add(keras.layers.Dense(num_classes, activation='softmax'))

    # Compile the model with specified optimizer, loss function, and metrics
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    loss = keras.losses.CategoricalCrossentropy()
    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=METRICS)  # Metrics defined below

    return model

# Define the EarlyStopping callback
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',  # or 'val_accuracy' depending on your preference
    patience=10,  # Number of epochs with no improvement after which training will be stopped
    restore_best_weights=True,  # Restore weights from the epoch with the best value
    verbose=1
)

# Custom metrics for model evaluation
METRICS = [
    keras.metrics.MeanSquaredError(name='Brier score'),
    keras.metrics.BinaryAccuracy(name='accuracy'),
    keras.metrics.Precision(name='precision'),
    keras.metrics.Recall(name='recall'),
    keras.metrics.AUC(name='auc'),
    keras.metrics.AUC(name='prc', curve='PR'),  # precision-recall curve
]

# Create the CNN model using make_model function

print('... defining model')
model = make_model(input_shape=input_shape,
                   num_classes=num_classes,
                   metrics=METRICS)

# Train the model with early stopping
print('... training model')
EPOCHS = 10
BATCH_SIZE = 512
history = model.fit(train_image_array, y_train_one_hot,
                    validation_data=(val_image_array, y_val_one_hot),
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE,
                    callbacks=[early_stopping],
                    verbose=0)

# Make predictions on validation and test datasets
print('... evaluating model')
y_pred_val = model.predict(val_image_array)
y_pred_val_labels = np.argmax(y_pred_val, axis=1)

y_pred = model.predict(test_image_array)
y_pred_labels = np.argmax(y_pred, axis=1)

# Print evaluation metrics for validation and test datasets
print(f'Validation\t | {print_metrics(val_image_label, y_pred_val)}')
print(f'Test:\t\t | {print_metrics(test_image_label, y_pred)}')

# Save the trained model
print('... saving model `emotion_classifier.h5` to ../models')
model.save('../models/emotion_classifier.h5')