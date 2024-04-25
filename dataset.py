import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from feature_extraction import FeatureExtractor
from utils import *

LABELS_CSV_FILE = 'csv/SEP-28k_fluencybank_labels_with_path.csv'
SAMPLES_LIMIT = 0
RANDOM_STATE = 42

TEST_SIZE = 0.1

def show_label_distribution(labels: dict) -> None:
    print('Total samples:', len(next(iter(labels.values()))))
    for label_name, values in labels.items():
        total = len(values)
        positives = np.count_nonzero(values)
        negatives = total - positives
        print(f'{label_name}: {positives}/{total} positives - {(positives / total) * 100:.2f}% positives, {(negatives / total) * 100:.2f}% negatives')

def resample_negatives(df: pd.DataFrame, column_name: str, th: int, random_state: int):
    df_negatives = df[df[column_name] < th]
    df_positives = df[df[column_name] >= th]

    df_negatives = df_negatives.sample(len(df_positives), random_state=random_state)
    
    return pd.concat([df_negatives, df_positives]).sample(frac=1, random_state=random_state)

def resample_positives(df: pd.DataFrame, column_name: str, th: int, random_state: int):
    df_negatives = df[df[column_name] < th]
    df_positives = df[df[column_name] >= th]

    df_positives = df_positives.sample(len(df_negatives), random_state=random_state, replace=True)

    return pd.concat([df_negatives, df_positives]).sample(frac=1, random_state=random_state)

def resample_positives_augmentation(df: pd.DataFrame, column_name: str, th: int, random_state: int):
    df_positives_original = df[df[column_name] >= th]
    df_negatives = df[df[column_name] < th]

    df_positives_resampled = df_positives_original.sample(len(df_negatives) - len(df_positives_original), random_state=random_state, replace=True)
    df_positives_resampled['Augment'] = True

    return pd.concat([df_negatives, df_positives_original, df_positives_resampled]).sample(frac=1, random_state=random_state)

def resample_positives_augmented_multilabel(df: pd.DataFrame, column_names: list, th: int, random_state: int):
    for name in column_names:
        df_positives_original = df[df[name] >= th]
        df_negatives = df[df[name] < th]

        df_positives_resampled = df_positives_original.sample(len(df_negatives) - len(df_positives_original), random_state=random_state, replace=True)
        df_positives_resampled['Augment'] = True

        df = pd.concat([df_negatives, df_positives_original, df_positives_resampled]).sample(frac=1, random_state=random_state)
    
    return df

def get_labels(df: pd.DataFrame, pos_labels: list, th=2):
    Y = {label: np.zeros(len(df)) for label in pos_labels}

    index = 0

    for _, row in df.iterrows():
        if 'Prolongation' in pos_labels and row['Prolongation'] >= th:
            Y['Prolongation'][index] = 1

        if 'Block' in pos_labels and row['Block'] >= th:
            Y['Block'][index] = 1

        if 'SoundRep' in pos_labels and row['SoundRep'] >= th:
            Y['SoundRep'][index] = 1

        if 'Repetition' in pos_labels and row['Repetition'] >= th:
            Y['Repetition'][index] = 1

        if 'Any' in pos_labels and (Y['Block'][index] or Y['Prolongation'][index] or Y['Repetition'][index]):
            Y['Any'][index] = 1

        index += 1

    return Y

def get_dataset(pos_labels: list, load_features: bool):
    # Load dataset csv and modify dataset path
    df = pd.read_csv(LABELS_CSV_FILE)

    # Clean up paths which do not exist and apply the limit
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
    df_train = resample_positives(df_train, pos_labels[0], 2, RANDOM_STATE)
    df_test = resample_negatives(df_test, pos_labels[0], 2, RANDOM_STATE)

    # Get training and validation features and labels
    if load_features:
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

    Y_train = get_labels(df_train, pos_labels, 2)
    Y_test = get_labels(df_test, pos_labels, 2)

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

    return (X_train, Y_train, X_test, Y_test)