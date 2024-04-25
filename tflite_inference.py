# Disable TF debugging logs
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import numpy as np
from dataset import get_dataset
from models import Model
from feature_extraction import FeatureExtractor

TFLITE_MODEL_PATH = 'trained_models/tflite/block.tflite'
TF_MODEL_PATH = 'trained_models/keras/block'
LABEL = 'Block'

X_train, Y_train, X_test, Y_test = get_dataset([LABEL], True)

sample = X_test[23]
sample_label = Y_test[LABEL][23]

# sample = FeatureExtractor.mfcc('silence.wav', n_fft=2048, hop=512, normalize=False, transpose=False)
# sample_label = 0

sample = (sample - np.mean(X_train)) / np.std(X_train)

tf_model = tf.keras.models.load_model(TF_MODEL_PATH)
output_tf = Model.predict_single(tf_model, sample)[LABEL].numpy()[0][0]

interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]
input_scale = input_details['quantization'][0]
input_zero_point = input_details['quantization'][1]
output_scale = output_details['quantization'][0]
output_zero_point = output_details['quantization'][1]

interpreter.set_tensor(input_details['index'], np.ndarray.astype(np.expand_dims(sample / input_scale + input_zero_point, axis=0), np.int8))

interpreter.invoke()

output_tensor = interpreter.get_tensor(output_details['index'])
output_tflite = (output_tensor[0][0] - output_zero_point) * output_scale

print('True label:', sample_label)
print('TFLite output:', output_tflite)
print('TF output:', output_tf)
