from models import CNN
from utils import show_label_distribution

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import keras
import tensorboard
import datetime

LABELS_CSV_FILE = 'SEP-28k_labels_with_path.csv'

TEST_SIZE = 0.2

pos_labels=['Prolongation', 'Repetition', 'Block']

def get_labels(df: pd.DataFrame, th=2):
    Y = {label: np.zeros(len(df)) for label in pos_labels}

    index = 0

    for _, row in df.iterrows():
        if row['Prolongation'] >= 2:
            Y['Prolongation'][index] = 1
        else:
            Y['Prolongation'][index] = 0

        if row['Block'] >= 2:
            Y['Block'][index] = 1
        else:
            Y['Block'][index] = 0

        if row['SoundRep'] >= 2 or row['WordRep'] >= 2:
            Y['Repetition'][index] = 1
        else:
            Y['Repetition'][index] = 0

        index += 1

    return Y


# Load dataset csv and modify dataset path
df = pd.read_csv(LABELS_CSV_FILE).head(10000)
# df['Path'] = DATASET_PATH + df['Path'].astype(str)

# Clean up paths which do not exist
df = df[df['Path'].apply(os.path.exists)]

# Train-test split
df_train, df_test = train_test_split(df, test_size=TEST_SIZE)

# Get training features and labels
X_train = np.load('features/mfcc_train.npy')
Y_train = get_labels(df_train, pos_labels)

# Get validation features and labels
X_test = np.load('features/mfcc_test.npy')
Y_test = get_labels(df_test, pos_labels)

# Print label distribution
print('Train')
show_label_distribution(Y_train)
print('Test')
show_label_distribution(Y_test)

# Build the model
model = CNN(pos_labels)

# Train the model
callbacks = [
    keras.callbacks.TensorBoard(
        log_dir = 'logs/fit/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
        histogram_freq = 1,
    )
]

model.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=256, epochs=50, callbacks=callbacks)

