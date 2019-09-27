#%%
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import math
import numpy as np

dataset, metaset = tfds.load('fashion_mnist', as_supervised=True, with_info=True)
train_dataset, test_dataset = dataset['train'], dataset['test']

#%%
print(train_dataset)

#%%
def normalize(images, labels):
    images = tf.cast(images, tf.float32)
    images = images / 255
    return images, labels

train_dataset = train_dataset.map(normalize)
test_dataset = test_dataset.map(normalize)

#%%
print(train_dataset)

#%%
l0 = tf.keras.layers.Flatten(input_shape=(28, 28, 1))
l1 = tf.keras.layers.Dense(512, activation=tf.nn.relu)
l1_1 = tf.keras.layers.Dense(256, activation=tf.nn.relu)
l1_2 = tf.keras.layers.Dense(128, activation=tf.nn.relu)
l1_3 = tf.keras.layers.Dense(64, activation=tf.nn.relu)
l1_4 = tf.keras.layers.Dense(32, activation=tf.nn.relu)
l2 = tf.keras.layers.Dense(10, activation=tf.nn.softmax)

model = tf.keras.Sequential()
model.add(l0)
model.add(l1)
model.add(l2)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# BATCH_SIZE = 32
# train_dataset = train_dataset.repeat().shuffle(60000).batch(BATCH_SIZE)
# test_dataset = test_dataset.batch(BATCH_SIZE)
# model.fit(train_dataset, epochs=5, steps_per_epoch=1875)


#%%
# BATCH_SIZE = 32
# train_dataset = train_dataset.repeat().shuffle(60000).batch(BATCH_SIZE)
# test_dataset = test_dataset.batch(BATCH_SIZE)

# model.fit(train_dataset)


#%%
train_dataset = train_dataset.batch(32)
test_dataset = test_dataset.batch(32)
model.fit(train_dataset, epochs=5)

#%%
test_loss, test_accuracy = model.evaluate(test_dataset)
print('Accuracy on test dataset:', test_accuracy)

#%%
