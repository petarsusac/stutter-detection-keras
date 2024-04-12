from models import *
from utils import *
from feature_extraction import FeatureExtractor

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import keras
import datetime

LABELS_CSV_FILE = 'csv/SEP-28k_fluencybank_labels_with_path.csv'
SAMPLES_LIMIT = 0
LOAD_FEATURES = True
RANDOM_STATE = 42

TEST_SIZE = 0.1

pos_labels=['Repetition']

def get_labels(df: pd.DataFrame, th=2):
    Y = {label: np.zeros(len(df)) for label in pos_labels}

    index = 0

    for _, row in df.iterrows():
        # if row['Prolongation'] >= th:
        #     Y['Prolongation'][index] = 1

        # if row['Block'] >= th:
        #     Y['Block'][index] = 1

        # if row['SoundRep'] >= th:
        #     Y['SoundRep'][index] = 1

        if row['Repetition'] >= th:
            Y['Repetition'][index] = 1

        # if Y['Block'][index] or Y['Prolongation'][index] or Y['Repetition'][index]:
        #     Y['Any'][index] = 1

        index += 1

    return Y


# Load dataset csv and modify dataset path
df = pd.read_csv(LABELS_CSV_FILE)

# Clean up paths which do not exist
df = df[df['Path'].apply(os.path.exists)]
if SAMPLES_LIMIT > 0:
    df = df.head(SAMPLES_LIMIT)

# Add 'Repetition' label by combining 'SoundRep' and 'WordRep'
df['Repetition'] = df[['SoundRep', 'WordRep']].max(axis=1)

# Mark the original samples to avoid augmentation
df['Augment'] = False

# Train-test split
df_train, df_test = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True)
# df_train = df[df['Show'] != 'FluencyBank']
# df_test = df[df['Show'] == 'FluencyBank']

# Resampling
df_train = resample_positives(df_train, 'Repetition', 2, RANDOM_STATE)
df_test = resample_negatives(df_test, 'Repetition', 2, RANDOM_STATE)

# Get training and validation features and labels
if LOAD_FEATURES:
    X_train = np.load('features/mfcc_train.npy')
    X_test = np.load('features/mfcc_test.npy')
else:
    feature_extractor = FeatureExtractor(df_train[['Path', 'Augment']])
    print('Generating training set features...')
    X_train = feature_extractor.extract(
        FeatureExtractor.mfcc, 
        'features/mfcc_train.npy',
        n_fft=2048,
        hop=512,
        normalize=False,
        transpose=False
    )

    feature_extractor = FeatureExtractor(df_test[['Path', 'Augment']])
    print('Generating validation set features...')
    X_test = feature_extractor.extract(
        FeatureExtractor.mfcc, 
        'features/mfcc_test.npy',
        n_fft=2048,
        hop=512,
        normalize=False,
        transpose=False
    )

Y_train = get_labels(df_train, 2)
Y_test = get_labels(df_test, 2)

# Check if the length of features and labels match
for labels in Y_train.values():
    assert X_train.shape[0] == labels.shape[0]
for labels in Y_test.values():
    assert X_test.shape[0] == labels.shape[0]

print('Training set shape:', X_train.shape)
print('Validation set shape:', X_test.shape)

# Normalization
train_mean = np.mean(X_train)
train_std = np.std(X_train)
print('Training set mean:', train_mean)
print('Training set std:', train_std)
X_train = (X_train - train_mean) / train_std
X_test = (X_test - train_mean) / train_std

# Print label distribution
print('Train')
show_label_distribution(Y_train)
print('Test')
show_label_distribution(Y_test)

# Build the model
model = ConvLSTM.create_model(pos_labels, input_shape=(X_train.shape[1], X_train.shape[2]))
model.summary()

# Train the model
callbacks = [
    keras.callbacks.TensorBoard(
        log_dir = 'logs/fit/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
        histogram_freq = 1,
    ),
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
]

model.fit(X_train, Y_train, 
          validation_data=(X_test, Y_test), 
          batch_size=256, 
          epochs=50, 
          callbacks=callbacks,
          verbose=2)

y_pred = Model.predict_batch(model, X_test)

for label in pos_labels:
    print(f'Confusion matrix ({label}):')
    print(confusion_matrix(Y_test[label], y_pred[label], th=0.5))

save_model(model, 'trained_models/keras/repetition', X_train.shape[1:])
