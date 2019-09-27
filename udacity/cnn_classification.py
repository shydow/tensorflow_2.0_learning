#%%
import tensorflow as tf
import tensorflow_datasets as tfds

dataset, metadata = tfds.load('fashion_mnist', as_supervised=True, with_info=True)
train_dataset, test_dataset = dataset['train'], dataset['test']

#%%
def normalize(image, label):
    image = tf.cast(image, tf.float32)
    image = image/255
    return image, label

train_dataset = train_dataset.map(normalize)
test_dataset = test_dataset.map(normalize)

#%%
l0 = tf.keras.layers.Conv2D(32, (3,3), padding='same', activation=tf.nn.relu, input_shape=(28, 28, 1))
l1 = tf.keras.layers.MaxPooling2D((2,2), 2)
l2 = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation=tf.nn.relu)
l3 = tf.keras.layers.MaxPooling2D((2,2), 2)
l4 = tf.keras.layers.Flatten()
l5 = tf.keras.layers.Dense(128, activation=tf.nn.relu)
l6 = tf.keras.layers.Dense(10, activation=tf.nn.softmax)

model = tf.keras.Sequential([l0, l1, l2, l3, l4, l5, l6])

#%%
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#%%
train_dataset = train_dataset.batch(32)
test_dataset = test_dataset.batch(32)

model.fit(train_dataset, epochs=5)

#%%
test_loss, test_accuracy = model.evaluate(test_dataset)
print('Accuracy on test dataset:', test_accuracy)

#%%
