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
        x = keras.layers.Conv2D(32, kernel_size=(1,3), activation='relu', padding='same')(x)
        x = keras.layers.Conv2D(32, kernel_size=(3,1), activation='relu', padding='same')(x)
        x = keras.layers.MaxPooling2D(pool_size=(2,2))(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Conv2D(32, kernel_size=(3,1), activation='relu', padding='same')(x)
        x = keras.layers.Conv2D(32, kernel_size=(1,3), activation='relu', padding='same')(x)
        x = keras.layers.MaxPooling2D(pool_size=(2,2))(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.GlobalMaxPooling2D()(x)
        x = keras.layers.Dense(128, activation='relu')(x)

        outputs = {label: keras.layers.Dense(1, activation='sigmoid', name=label)(x) for label in output_labels}

        self.keras_model = keras.Model(inputs=input, outputs=outputs)

        self.keras_model.compile(
            optimizer=keras.optimizers.Adam(1e-3),
            loss={label: 'binary_crossentropy' for label in output_labels},
            metrics=[keras.metrics.BinaryAccuracy(threshold=0.5)],
        )

class GRUNet(Model):
    def __init__(self, output_labels: list, input_shape: tuple = (92, 40), timestep=40) -> None:
        super().__init__()
        
        input = keras.Input(shape=input_shape)

        x = keras.layers.Reshape((-1, timestep))(input)

        x = keras.layers.Bidirectional(keras.layers.GRU(32, return_sequences=True))(x)
        x = keras.layers.Bidirectional(keras.layers.GRU(32, return_sequences=False))(x)

        x = keras.layers.Dropout(0.2)(x)
        
        outputs = {label: keras.layers.Dense(1, activation='sigmoid', name=label)(x) for label in output_labels}

        self.keras_model = keras.Model(inputs=input, outputs=outputs)

        self.keras_model.compile(
            optimizer=keras.optimizers.Adam(5e-4),
            loss={label: 'binary_crossentropy' for label in output_labels},
            metrics=[keras.metrics.BinaryAccuracy(threshold=0.5)],
        )

class ConvGRU(Model):
    def __init__(self, output_labels: list, input_shape: tuple = (40, 92)) -> None:
        super().__init__()

        input = keras.Input(shape=input_shape)

        x = keras.layers.Reshape((*input_shape, 1))(input)

        x = keras.layers.Conv2D(32, kernel_size=(1, 3), padding='same', kernel_initializer=keras.initializers.RandomNormal(stddev=0.05), 
                     bias_initializer=keras.initializers.Zeros())(x)
        x = keras.layers.LeakyReLU(alpha=0.01)(x)
        x = keras.layers.Conv2D(32, kernel_size=(1, 3), padding='same', kernel_initializer=keras.initializers.RandomNormal(stddev=0.05), 
                     bias_initializer=keras.initializers.Zeros())(x)
        x = keras.layers.LeakyReLU(alpha=0.01)(x)
        x = keras.layers.Conv2D(32, kernel_size=(1, 1), padding='same', kernel_initializer=keras.initializers.RandomNormal(stddev=0.05), 
                     bias_initializer=keras.initializers.Zeros())(x)
        x = keras.layers.LeakyReLU(alpha=0.01)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.MaxPooling2D(pool_size=(1, 2))(x)
        x = keras.layers.Conv2D(32, kernel_size=(3, 1), padding='same', kernel_initializer=keras.initializers.RandomNormal(stddev=0.05), 
                     bias_initializer=keras.initializers.Zeros())(x)
        x = keras.layers.LeakyReLU(alpha=0.01)(x)
        x = keras.layers.Conv2D(32, kernel_size=(3, 1), padding='same', kernel_initializer=keras.initializers.RandomNormal(stddev=0.05), 
                     bias_initializer=keras.initializers.Zeros())(x)
        x = keras.layers.LeakyReLU(alpha=0.01)(x)
        x = keras.layers.Conv2D(32, kernel_size=(1, 1), padding='same', kernel_initializer=keras.initializers.RandomNormal(stddev=0.05), 
                     bias_initializer=keras.initializers.Zeros())(x)
        x = keras.layers.LeakyReLU(alpha=0.01)(x)
        x = keras.layers.MaxPooling2D(pool_size=(2, 1))(x)
        x = keras.layers.Conv2D(64, kernel_size=(3, 3), padding='same', kernel_initializer=keras.initializers.RandomNormal(stddev=0.05), 
                     bias_initializer=keras.initializers.Zeros())(x)
        x = keras.layers.LeakyReLU(alpha=0.01)(x)
        x = keras.layers.Conv2D(32, kernel_size=(1, 1), padding='same', kernel_initializer=keras.initializers.RandomNormal(stddev=0.05), 
                     bias_initializer=keras.initializers.Zeros())(x)
        x = keras.layers.LeakyReLU(alpha=0.01)(x)
        x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)

        x = keras.layers.Permute((2, 1, 3))(x)
        x = keras.layers.TimeDistributed(keras.layers.Flatten())(x)

        x = keras.layers.Bidirectional(keras.layers.GRU(128, return_sequences=False))(x)

        x = keras.layers.Dense(64)(x)
        
        outputs = {label: keras.layers.Dense(1, activation='sigmoid', name=label)(x) for label in output_labels}

        self.keras_model = keras.Model(inputs=input, outputs=outputs)

        self.keras_model.compile(
            optimizer=keras.optimizers.Adam(5e-4),
            loss={label: 'binary_crossentropy' for label in output_labels},
            metrics=[keras.metrics.F1Score(threshold=0.5)],
        )

class LSTM(Model):
    def __init__(self, output_labels: list, input_shape: tuple = (40, 92)) -> None:
        super().__init__()

        input = keras.Input(shape=input_shape)

        x = keras.layers.LSTM(64)(input)

        outputs = {label: keras.layers.Dense(1, activation='sigmoid', name=label)(x) for label in output_labels}

        self.keras_model = keras.Model(inputs=input, outputs=outputs)

        self.keras_model.compile(
            optimizer=keras.optimizers.Adam(0.01),
            loss={label: 'binary_crossentropy' for label in output_labels},
            metrics=[keras.metrics.BinaryAccuracy(threshold=0.5)],
        )
    
class ResNetLSTM(Model):
    def resblock(x, filter_sizes):
        res_conn = x

        res_conn = keras.layers.Conv2D(filter_sizes[0], kernel_size=(3,3))(res_conn)
        res_conn = keras.layers.BatchNormalizetion()(res_conn)

        x = keras.layers.Conv2D(filter_sizes[0], kernel_size=(3,3))(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)

        x = keras.layers.Conv2D(filter_sizes[1], kernel_size=(3,3))(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)

        x = keras.layers.Conv2D(filter_sizes[2], kernel_size=(3,3))(x)
        x = keras.layers.BatchNormalization()(x)

        x = keras.layers.Add([x, res_conn])
        
        x = keras.layers.ReLU()(x)

        return x


    def __init__(self, output_labels: list, input_shape: tuple) -> None:
        super().__init__()

        input = keras.Input(shape=input_shape)

        x = keras.layers.Conv2D(64, kernel_size=(7,7))(input)
        x = ResNetLSTM.resblock(x, (32, 64, 64))
        x = ResNetLSTM.resblock(x, (64, 128, 128))
        x = ResNetLSTM.resblock(x, (128, 128, 128))
        x = ResNetLSTM.resblock(x, (128, 64, 64))
        x = ResNetLSTM.resblock(x, (64, 64, 32))
        x = ResNetLSTM.resblock(x, (32, 16, 16))
        x = keras.layers.Flatten()(x)
        x = keras.layers.Bidirectional(keras.layers.LSTM(512, return_sequences=True))(x)
        x = keras.layers.Bidirectional(keras.layers.LSTM(512))(x)

        outputs = {label: keras.layers.Dense(1, activation='sigmoid', name=label)(x) for label in output_labels}

        self.keras_model = keras.Model(inputs=input, outputs=outputs)

        self.keras_model.compile(
            optimizer=keras.optimizers.Adam(1e-4),
            loss={label: 'binary_crossentropy' for label in output_labels},
            metrics=[keras.metrics.BinaryAccuracy(threshold=0.5)],
        )


