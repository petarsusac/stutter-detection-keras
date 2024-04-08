import numpy as np

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
