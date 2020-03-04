def evaluate_model(model, test_data, test_labels):
    print("Evaluating model on {} examples".format(test_data.shape[0]))
    model_loss, model_accuracy = model.evaluate(test_data, test_labels, verbose=2)
    return dict(loss=model_loss, accuracy=model_accuracy)
