from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
tf.keras.backend.set_floatx('float64')
mnist = tf.keras.datasets.mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train, X_test = X_train/255.0, X_test/255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
])

predictions = model(X_train[:1]).numpy()
print(predictions)
print(tf.nn.softmax(predictions).numpy())

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
print(loss_fn(y_train[:1], predictions).numpy())

model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
model.fit(X_train, y_train, epochs=5)

model.evaluate(X_test, y_test, verbose=2)