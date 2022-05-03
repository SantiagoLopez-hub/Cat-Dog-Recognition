import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import numpy as np

print(tf.__version__)

# Pre-processing training set
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

training_set = train_datagen.flow_from_directory(
    'dataset/training_set',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary')

# Pre-processing testing set
test_datagen = ImageDataGenerator(rescale=1. / 255)

testing_set = test_datagen.flow_from_directory(
    'dataset/test_set',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary')

# Initialising Convolutional Neural Network
cnn = tf.keras.models.Sequential()

# Convolution
cnn.add(tf.keras.layers.Conv2D(
    filters=32,
    kernel_size=3,
    activation='relu',
    input_shape=[64, 64, 3]))

# Pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Add second convolutional layer
cnn.add(tf.keras.layers.Conv2D(
    filters=32,
    kernel_size=3,
    activation='relu'))

cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Flattening
cnn.add(tf.keras.layers.Flatten())

# Full connection
cnn.add(tf.keras.layers.Dense(
    units=128,
    activation='relu'))

# Output layer
cnn.add(tf.keras.layers.Dense(
    units=1,
    activation='sigmoid'))

# Compile and train CNN
cnn.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy'])

cnn.fit(x=training_set, validation_data=testing_set, epochs=25)

# Make prediction
test_image = image.load_img(
    'dataset/single_prediction/cat_or_dog_1.jpg',
    target_size=(64, 64))

# Convert PIL to array
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)

result = cnn.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'

print(prediction)
