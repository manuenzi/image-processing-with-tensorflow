from __future__ import absolute_import, division, print_function, unicode_literals
import os
import tensorflow as tf
from tensorflow import keras
import create_model as crm
import compile_model as cpm
import train_model as tm
import evaluate_model as evm
import execute_model as exm
import numpy as np
import matplotlib.pyplot as plt

fashion_mnist = keras.datasets.fashion_mnist
(trainval_images, trainval_labels), (test_images, test_labels) = fashion_mnist.load_data()
random_index = np.random.permutation(trainval_images.shape[0])

val_images = trainval_images[random_index[:round(0.05*trainval_images.shape[0])], :, :]
val_labels = trainval_labels[random_index[:round(0.05*trainval_images.shape[0])]]

train_images = trainval_images[random_index[round(0.05*trainval_images.shape[0]):], :, :]
train_labels = trainval_labels[random_index[round(0.05*trainval_images.shape[0]):]]
print(trainval_images.shape[0])
print(train_images.shape[0])
print(val_images.shape[0])
class_names = ['Tshirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
#print("Shape of training data {}".format(train_images.shape))

train_images = train_images / 255.0
val_images = val_images / 255.0
test_images = test_images / 255.0

#plt.figure(figsize=(10, 10))
#for i in range(25):
#    plt.subplot(5, 5, i+1)
#    plt.xticks([])
#    plt.yticks([])
#    plt.grid(False)
#    plt.imshow(train_images[i], cmap=plt.cm.binary)
#    plt.xlabel(class_names[train_labels[i]])
#plt.show()

model = crm.create_model()
model = cpm.compile_model(model)
print(model.summary())

checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)

model = tm.train_model(model, train_images, train_labels, val_images, val_labels, 5, callbacks=[cp_callback])
eval_dict = evm.evaluate_model(model, test_images, test_labels)

print("Model loss equals {:.3f} and model accuracy equals {:.3f}".format(eval_dict["loss"], eval_dict["accuracy"]))


untrained_model = crm.create_model()
untrained_model = cpm.compile_model(untrained_model)
print(untrained_model.summary())

untrained_eval_dict = evm.evaluate_model(untrained_model, test_images, test_labels)

print("Untrained loss {:.3f} accuracy {:.3f}".format(untrained_eval_dict["loss"], untrained_eval_dict["accuracy"]))

untrained_model.load_weights(checkpoint_path)
untrained_eval_dict = evm.evaluate_model(untrained_model, test_images, test_labels)

print("new Untrained loss {:.3f} accuracy {:.3f}".format(untrained_eval_dict["loss"], untrained_eval_dict["accuracy"]))

checkpoint_path = "training_2/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, verbose=1, save_weights_only=True, period=5)
model = crm.create_model()
model = cpm.compile_model(model)
model.save_weights(checkpoint_path.format(epoch=0))

model = tm.train_model(model, train_images, train_labels, val_images, val_labels, 5, callbacks=[cp_callback])

latest = tf.train.latest_checkpoint(checkpoint_dir)
model.save_weights('./checkpoints/my_checkpoint')

model = crm.create_model()
model = cpm.compile_model(model)
model.load_weights('./checkpoints/my_checkpoint')

eval_dict = evm.evaluate_model(model, test_images, test_labels)

print("Restored model accuracy : {:5.2f}%".format(100*eval_dict["accuracy"]))

model = crm.create_model()
model = cpm.compile_model(model)

model = tm.train_model(model, train_images, train_labels, val_images, val_labels, 3)

model.save('saved_model/my_model')
print("Restoring model")
new_model = tf.keras.models.load_model('saved_model/my_model')
new_model.summary()

model.save('saved_model/my_modelh5.h5')
print("Restoring model H5")
new_model = tf.keras.models.load_model('saved_model/my_modelh5.h5')
new_model.summary()
quit()

temp_res = exm.execute_model(model, test_images)
predictions = temp_res["predictions"]
probability_model = temp_res["model"]

print("predicted probabilities {}".format(predictions[17]))
print("true label {} predicted as label {}".format(test_labels[17], np.argmax(predictions[17])))

def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array, true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap=plt.cm.binary)
    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'
    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label], 100*np.max(predictions_array),
                                         class_names[true_label], color=color))

def plot_value_array(i, predictions_array, true_label):


    predictions_array, true_label = predictions_array, true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')
i = 0
plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1, 2, 2)
plot_value_array(i, predictions[i], test_labels)
plt.show()

i = 17
plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1, 2, 2)
plot_value_array(i, predictions[i], test_labels)
plt.show()

num_rows = 5
num_cols = 3
num_images = num_rows * num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))

for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(i, predictions[i], test_labels, test_images)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()

img = test_images[9]
print("Image shape {}".format(img.shape))
img = (np.expand_dims(img, 0))
print("Expanded image shape {}".format(img.shape))

predictions_single = probability_model.predict(img)

print("Predictions on one image {}".format(predictions_single))

plot_value_array(1, predictions_single[0], test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)
