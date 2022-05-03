import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator


print(tf.__version__)


train_datagen = ImageDataGenerator(
	rescale=1./255,
	shear_range=0.2,
	zoom_range=0.2,
	horizontal_flip=True)

