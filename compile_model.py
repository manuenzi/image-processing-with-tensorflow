from tensorflow import keras


lr_schedule = keras.optimizers.schedules.InverseTimeDecay(0.001, decay_steps=1000, decay_rate=1,
                                                          staircase=False)


def get_optimizer():
    return keras.optimizers.Adam(lr_schedule)


def compile_model(model, optimizer=None):

    if optimizer is None:
        optimizer = get_optimizer()
        print("Compiling model with 'adam' optimizer 'sparse categorical cross entropy' loss and 'accuracy' metric")
    else:
        print("Compiling model with {} optimizer 'sparse categorical cross entropy' loss and 'accuracy' "
              "metric".format(optimizer))
    model.compile(optimizer=optimizer,
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model
