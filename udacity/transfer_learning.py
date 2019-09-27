#%%
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds

import PIL as pil
import numpy as np

import logging

#%%
CLASSIFIER_URL ="https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/2"
IMAGE_SHAPE = 224

model = tf.keras.Sequential([
    hub.KerasLayer(CLASSIFIER_URL, input_shape=(IMAGE_SHAPE, IMAGE_SHAPE, 3))
])

#%%
grace_hopper = tf.keras.utils.get_file('image.jpg','https://storage.googleapis.com/download.tensorflow.org/example_images/grace_hopper.jpg')
grace_hopper = pil.Image.open(grace_hopper).resize((IMAGE_SHAPE, IMAGE_SHAPE))
grace_hopper 

#%%
grace_hopper = np.array(grace_hopper)/255.0
grace_hopper.shape

#%%
result = model.predict(grace_hopper[np.newaxis, ...])
result.shape

#%%
predicted_class = np.argmax(result[0], axis=-1)
predicted_class

#%%
splits = tfds.Split.ALL.subsplit(weighted=(80, 20))
(train_dataset, val_dataset), metaset = tfds.load('cats_vs_dogs', as_supervised=True, with_info=True, split=splits)


#%%
def normalize(image, label):
    image = tf.cast(image, tf.float32)
    image = image/255
    return image, label

train_dataset = train_dataset.map(normalize)
test_dataset = test_dataset.map(normalize)
#%%
