import tensorflow as tf

def compute_loss(model, x, y):
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    y_pred = model.model(x)
    return loss_fn(y, y_pred).numpy()
