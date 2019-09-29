#%%
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import os
from datetime import datetime

#%%
splits = tfds.Split.TRAIN.subsplit([70, 30])
tfds.list_builders()
(training_dataset, val_dataset), dataset_info = tfds.load('tf_flowers', with_info=True, as_supervised=True, split=splits)


#%%
BATCH_SIZE = 32
IMG_SHAPE = 224

def image_process(image, label):
    image = tf.image.resize(image, (IMG_SHAPE, IMG_SHAPE))
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

# training_dataset = training_dataset.map(lambda image, label: (tf.image.resize(image, [IMG_SHAPE, IMG_SHAPE])/255.0, label)).shuffle(1024).batch(BATCH_SIZE)
# val_dataset = val_dataset.map(lambda image, label: (tf.image.resize(image, [IMG_SHAPE, IMG_SHAPE])/255.0, label)).batch(BATCH_SIZE)
training_dataset = training_dataset.map(image_process).shuffle(1024).batch(BATCH_SIZE)
val_dataset = val_dataset.map(image_process).batch(BATCH_SIZE)

#%%
model = tf.keras.Sequential()
CLASSIFIER_URL ="https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/2"
hub_layer = hub.KerasLayer(CLASSIFIER_URL, input_shape=(IMG_SHAPE, IMG_SHAPE, 3))
hub_layer.trainable = False
model.add(hub_layer)
model.add(tf.keras.layers.Dense(5, activation = tf.nn.softmax))

#%%
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

#%%
logdir="logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

#%%
EPOCHS = 10
history = model.fit_generator(
    training_dataset,
    epochs=EPOCHS,
    validation_data=val_dataset,
    callbacks=[tensorboard_callback]
)

#%%
acc = history.history['acc']
loss = history.history['loss']

val_acc = history.history['val_acc']
val_loss = history.history['val_loss']

epochs_range = range(EPOCHS)
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()