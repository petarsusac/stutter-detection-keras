import numpy as np
import sklearn
import pandas as pd
import tensorflow as tf

def confusion_matrix(y_true, y_pred, th=0.5):
    y_pred = (y_pred > th).astype(np.float32)
    return sklearn.metrics.confusion_matrix(y_true, y_pred, normalize='true')

def save_model(model, path: str, input_shape: tuple):
    run_model = tf.function(lambda x: model(x))
    concrete_func = run_model.get_concrete_function(tf.TensorSpec([1, *input_shape], model.inputs[0].dtype))
    model.save(path, save_format="tf", signatures=concrete_func)