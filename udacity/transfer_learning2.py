#%%
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub

import matplotlib as plt

splits = tfds.Split.ALL.subsplit(weighted=(80, 20))
(train_dataset, test_dataset), metaset = tfds.load('cats_vs_dogs', as_supervised=True, with_info=True, split=splits)


#%%
train_dataset = tf.image.resize_image(train_dataset, (244.0,244.0))

#%%
def normalize(image, label):
    image = tf.image.resize_image(image, (224,224))
    image = tf.cast(image, tf.float32)
    image = image/255
    return image, label

train_dataset = train_dataset.map(normalize)
test_dataset = test_dataset.map(normalize)

#%%
CLASSIFIER_URL ="https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/2"
IMAGE_SHAPE = 224
feature_layer = hub.KerasLayer(CLASSIFIER_URL, input_shape=(IMAGE_SHAPE, IMAGE_SHAPE, 3))
feature_layer.trainable = False

model = tf.keras.Sequential([
    feature_layer,
    tf.keras.layers.Dense(2, activation=tf.nn.softmax)
])

#%%
model.compile(
  optimizer='adam',
  loss='sparse_categorical_crossentropy',
  metrics=['accuracy'])

model.summary()
#%%
EPOCHS = 6
train_dataset = train_dataset.batch(32)
test_dataset = test_dataset.batch(32)

history = model.fit_generator(
    train_dataset,
    epochs=EPOCHS
)
#%%
acc = history.history['acc']
val_acc = history.history['val_acc']

loss = history.history['loss']
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