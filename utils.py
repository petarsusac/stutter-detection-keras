import numpy as np

def show_label_distribution(labels: dict) -> None:
    print('Total samples:', len(next(iter(labels.values()))))
    for label_name, values in labels.items():
        total = len(values)
        positives = np.count_nonzero(values)
        negatives = total - positives
        print(f'{label_name}: {positives}/{total} positives - {(positives / total) * 100:.2f}% positives, {(negatives / total) * 100:.2f}% negatives')
