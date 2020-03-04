from tensorflow import keras
from tensorflow.keras import regularizers

def create_model():

    print("Creating a keras sequential model!")
    model = keras.Sequential()
    print("Adding flatten layer with input_shape (28, 28)"),
    model.add(keras.layers.Flatten(input_shape=(28, 28))),
    print("Adding dense layer with activation 'relu' with regularizer 'L2'"),
    model.add(keras.layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001))),
    print("Adding dropout layer with drop rate 0.3"),
    model.add(keras.layers.Dropout(0.3))
    print("Adding dense output layer with 10 units"),
    model.add(keras.layers.Dense(10))

    return model
