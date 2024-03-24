import tensorflow as tf
from tensorflow import keras

class Model:
    keras_model: keras.Model

    def __init__(self) -> None:
        pass

    def fit(self, *args, **kwargs) -> keras.callbacks.History:
        return self.keras_model.fit(*args, **kwargs)
    
    def predict_single(self, input):
        return self.keras_model(tf.constant([input]), training=False)
    
    def predict_batch(self, input_array):
        return self.keras_model.predict(input_array)

class CNN(Model):
    def __init__(self, output_labels: list, input_shape: tuple = (13, 43)) -> None:
        super().__init__()

        input = keras.Input(shape=(input_shape[0], input_shape[1]))

        x = keras.layers.Reshape((input_shape[0], input_shape[1], 1))(input)
        x = keras.layers.Conv2D(32, kernel_size=(1,3), activation='relu')(x)
        x = keras.layers.MaxPooling2D(pool_size=(2,2))(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Conv2D(64, kernel_size=(1,3), activation='relu')(x)
        x = keras.layers.MaxPooling2D(pool_size=(2,2))(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Conv2D(64, kernel_size=(1,3), activation='relu')(x)
        x = keras.layers.MaxPooling2D(pool_size=(2,2))(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.GlobalMaxPooling2D()(x)
        x = keras.layers.Dense(128, activation='relu')(x)

        outputs = {label: keras.layers.Dense(1, activation='sigmoid', name=label)(x) for label in output_labels}

        self.keras_model = keras.Model(inputs=input, outputs=outputs)

        self.keras_model.compile(
            optimizer=keras.optimizers.Adam(5e-4),
            loss={label: 'binary_crossentropy' for label in output_labels},
            metrics=[keras.metrics.BinaryAccuracy(threshold=0.5)],
        )

    




