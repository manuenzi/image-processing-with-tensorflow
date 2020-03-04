def train_model(model, train_data, train_labels, val_data, val_labels, epochs, callbacks=None):
    print("Fitting model with {} epochs and {} training examples".format(epochs, train_data.shape[0]))
    model.fit(train_data, train_labels, validation_data=(val_data, val_labels), epochs=epochs, callbacks=callbacks)
    return model
