from models import *
from utils import *
from feature_extraction import FeatureExtractor
from dataset import get_dataset

import keras
import datetime

pos_labels = ['Block']

X_train, Y_train, X_test, Y_test = get_dataset(pos_labels, True)

# Build the model
model = ConvLSTM.create_model(pos_labels, input_shape=(X_train.shape[1], X_train.shape[2]))
model.summary()

# Train the model
callbacks = [
    keras.callbacks.TensorBoard(
        log_dir = 'logs/fit/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
        histogram_freq = 1,
    ),
    # keras.callbacks.EarlyStopping(
    #     monitor='val_loss',
    #     patience=10,
    #     restore_best_weights=True
    # )
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

save_model(model, 'trained_models/keras/block', X_train.shape[1:])
