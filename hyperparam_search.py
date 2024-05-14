import keras
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

from models import *
from dataset import get_dataset

output_labels = ['Prolongation']

X_train, Y_train, X_test, Y_test = get_dataset(output_labels, False)

if len(output_labels) == 1:
    Y_train = Y_train[output_labels[0]]

model = KerasClassifier(
    build_fn=ConvLSTM.create_model,
    output_labels=output_labels,
    input_shape=X_train.shape[1:]
)

param_grid = {
    'num_timesteps': [12, 32, 64, 128],
    'num_conv_layers': [2, 3, 4],
    'num_conv_filters': [(32, 64, 64, 64), (32, 32, 32, 32)],
    'kern_size': [(3,3), (1,3), (3,1), (5,1)],
    'num_lstm_layers': [1, 2],
    'hidden_layer_neurons': [0, 64]
}

grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=2)
grid_result = grid.fit(X_train, Y_train, verbose=0)

print("Best parameters:", grid_result.best_params_)
print("Best accuracy:", grid_result.best_score_)

