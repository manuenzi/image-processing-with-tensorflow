from tensorflow import keras


def execute_model(model, new_data):
    probability_model = keras.Sequential([model, keras.layers.Softmax()])
    print("Predicting labels on {} new data".format(new_data.shape[0]))
    predictions = probability_model.predict(new_data)
    return dict(model=probability_model, predictions=predictions)
