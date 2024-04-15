import keras
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

from models import *
from dataset import get_dataset

output_labels = ['Block']

X_train, Y_train, X_test, Y_test = get_dataset(output_labels)

if len(output_labels) == 1:
    Y_train = Y_train[output_labels[0]]

model = KerasClassifier(
    build_fn=ConvLSTM.create_model,
    output_labels=output_labels,
    input_shape=X_train.shape[1:]
)

param_grid = {

}

grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=2)
grid_result = grid.fit(X_train, Y_train)

print("Best parameters:", grid_result.best_params_)
print("Best accuracy:", grid_result.best_score_)

