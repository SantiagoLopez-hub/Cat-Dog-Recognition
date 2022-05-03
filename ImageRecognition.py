import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator


print(tf.__version__)


# Pre processing training set
train_datagen = ImageDataGenerator(
	rescale=1./255,
	shear_range=0.2,
	zoom_range=0.2,
	horizontal_flip=True)

training_set = train_datagen.flow_from_directory(
	'dataset/training_set',
	target_size=(64, 64),
	batch_size=32,
	class_mode='binary')


# Pre processing testing set
test_datagen = ImageDataGenerator(rescale=1./255)

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
	input_shape=[64,64,3]))


# Pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
