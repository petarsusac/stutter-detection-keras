from models import CNN, GRUNet, ConvGRU
from utils import show_label_distribution_single
from feature_extraction import FeatureExtractor

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import keras
import datetime

LABELS_CSV_FILE = 'SEP-28k_labels_with_path.csv'
SAMPLES_LIMIT = 0
LOAD_FEATURES = True

TEST_SIZE = 0.2
RANDOM_SEED = 42

def get_single_label(df: pd.DataFrame, th=2):
    return df.apply(lambda row: row['Block'] >= th, axis='columns').to_numpy(dtype=np.float32)

# Load dataset csv
df = pd.read_csv(LABELS_CSV_FILE)

df = df[df['Path'].apply(os.path.exists)]
if SAMPLES_LIMIT > 0:
    df = df.head(SAMPLES_LIMIT)

# Train-test split
df_train, df_test = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_SEED)

# Build the model
model = CNN(['Block'], input_shape=(13,43))
model.keras_model.summary()

input("Waiting to continue...")

# Get training and validation features and labels
if LOAD_FEATURES:
    X_train = np.load('features/mfcc_train.npy')
    X_test = np.load('features/mfcc_test.npy')
else:
    feature_extractor = FeatureExtractor(df_train['Path'])
    print('Generating training set features...')
    X_train = feature_extractor.extract(
        FeatureExtractor.mfcc, 
        'features/mfcc_train.npy',
    )

    feature_extractor = FeatureExtractor(df_test['Path'])
    print('Generating validation set features...')
    X_test = feature_extractor.extract(
        FeatureExtractor.mfcc, 
        'features/mfcc_test.npy',
    )

Y_train = get_single_label(df_train, th=1)
Y_test = get_single_label(df_test, th=1)

assert Y_train.shape[0] == X_train.shape[0]
assert Y_test.shape[0] == X_test.shape[0]

print('Training set shape:', X_train.shape)
print('Validation set shape:', X_test.shape)

# Print label distribution
print('Train')
show_label_distribution_single(Y_train)
print('Test')
show_label_distribution_single(Y_test)

input("Waiting to continue...")

# Train the model
callbacks = [
    keras.callbacks.TensorBoard(
        log_dir = 'logs/fit/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
        histogram_freq = 1,
    )
]

model.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=256, epochs=100, verbose=2)

