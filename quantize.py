import tensorflow as tf
import numpy as np
import os
import subprocess
import netron

SAVED_MODEL_PATH = 'trained_models/keras/repetition'
TFLITE_MODEL_PATH = 'trained_models/tflite/repetition.tflite'
CPP_FILE_PATH = 'trained_models/cpp/repetition.cpp'
MODEL_NAME = 'model_repetition'
DATASET_PATH = 'features/mfcc_train.npy'

dataset = np.load(DATASET_PATH)

def representative_dataset():
    LIMIT = 500
    for i in range(LIMIT):
        yield [dataset[i]]

converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL_PATH)

converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.target_spec.supported_types = [tf.int8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
converter.experimental_new_quantizer = False

converter.representative_dataset = tf.lite.RepresentativeDataset(representative_dataset)

model_tflite = converter.convert()

with open(TFLITE_MODEL_PATH, "wb") as file:
    file.write(model_tflite)

print('Conversion successful. Model size:', os.path.getsize(TFLITE_MODEL_PATH))

subprocess.run(f'xxd -i {TFLITE_MODEL_PATH} > {CPP_FILE_PATH}', shell=True)

with open(CPP_FILE_PATH, 'r') as file:
    cpp_file = file.read()

cpp_file = cpp_file.replace(TFLITE_MODEL_PATH.replace('/','_').replace('.','_'), MODEL_NAME)
cpp_file = cpp_file.replace('unsigned', 'const unsigned')
cpp_file = f'#include \"{MODEL_NAME}.hpp\"\nalignas(4) ' + cpp_file

with open(CPP_FILE_PATH, 'w') as file:
    file.write(cpp_file)

print('Successfully generated CPP file:', CPP_FILE_PATH)

netron.start(TFLITE_MODEL_PATH)
