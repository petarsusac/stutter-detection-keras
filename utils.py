import numpy as np
import sklearn
import pandas as pd

def show_label_distribution(labels: dict) -> None:
    print('Total samples:', len(next(iter(labels.values()))))
    for label_name, values in labels.items():
        total = len(values)
        positives = np.count_nonzero(values)
        negatives = total - positives
        print(f'{label_name}: {positives}/{total} positives - {(positives / total) * 100:.2f}% positives, {(negatives / total) * 100:.2f}% negatives')

def show_label_distribution_single(labels: np.array) -> None:
    print('Total samples:', len(labels))
    total = len(labels)
    positives = np.count_nonzero(labels)
    negatives = total - positives
    print(f'{positives}/{total} positives - {(positives / total) * 100:.2f}% positives, {(negatives / total) * 100:.2f}% negatives')

def confusion_matrix(y_true, y_pred, th=0.5):
    y_pred = (y_pred > th).astype(np.float32)
    return sklearn.metrics.confusion_matrix(y_true, y_pred)

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