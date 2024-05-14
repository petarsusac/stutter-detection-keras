import tensorflow as tf
from tensorflow import keras

class Model:    
    def predict_single(keras_model, input):
        return keras_model(tf.constant([input]), training=False)
    
    def predict_batch(keras_model, input_array):
        return keras_model.predict(input_array)

class CNN(Model):
    def create_model(output_labels: list, input_shape: tuple = (13, 43)) -> keras.Model:
        input = keras.Input(shape=input_shape)

        x = keras.layers.Reshape((input_shape[0], input_shape[1], 1))(input)
        x = keras.layers.Conv2D(32, kernel_size=(3,3), activation='relu', padding='same')(x)
        x = keras.layers.Conv2D(32, kernel_size=(3,3), activation='relu', padding='same')(x)
        x = keras.layers.MaxPooling2D(pool_size=(2,2))(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Conv2D(64, kernel_size=(3,3), activation='relu', padding='same')(x)
        x = keras.layers.Conv2D(64, kernel_size=(3,3), activation='relu', padding='same')(x)
        x = keras.layers.MaxPooling2D(pool_size=(2,2))(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(128, activation='relu')(x)

        outputs = {label: keras.layers.Dense(1, activation='sigmoid', name=label)(x) for label in output_labels}

        keras_model = keras.Model(inputs=input, outputs=outputs)

        keras_model.compile(
            optimizer=keras.optimizers.Adam(5e-4),
            loss={label: 'binary_crossentropy' for label in output_labels},
            metrics=[keras.metrics.BinaryAccuracy(threshold=0.5),
                     keras.metrics.Precision(),
                     keras.metrics.Recall()]
        )

        return keras_model

class GRUNet(Model):
    def create_model(output_labels: list, input_shape: tuple = (92, 40), timestep=40) -> keras.Model:
        input = keras.Input(shape=input_shape)
        x = keras.layers.Permute((2, 1))(input)

        x = keras.layers.Reshape((-1, timestep))(input)

        x = keras.layers.Bidirectional(keras.layers.GRU(32, return_sequences=True))(x)
        x = keras.layers.Bidirectional(keras.layers.GRU(32, return_sequences=False))(x)

        x = keras.layers.Dropout(0.2)(x)
        
        outputs = {label: keras.layers.Dense(1, activation='sigmoid', name=label)(x) for label in output_labels}

        keras_model = keras.Model(inputs=input, outputs=outputs)

        keras_model.compile(
            optimizer=keras.optimizers.Adam(5e-4),
            loss={label: 'binary_crossentropy' for label in output_labels},
            metrics=[keras.metrics.BinaryAccuracy(threshold=0.5)],
        )

        return keras_model

class ConvGRU(Model):
    def create_model(output_labels: list, input_shape: tuple = (40, 92)) -> keras.Model:
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

        keras_model = keras.Model(inputs=input, outputs=outputs)

        keras_model.compile(
            optimizer=keras.optimizers.Adam(1e-4),
            loss={label: 'binary_crossentropy' for label in output_labels},
            metrics=[keras.metrics.BinaryAccuracy(threshold=0.5),
                     keras.metrics.Precision(),
                     keras.metrics.Recall()],
        )

        return keras_model

class LSTM(Model):
    def create_model(output_labels: list, input_shape: tuple = (40, 92)) -> keras.Model:
        input = keras.Input(shape=input_shape)

        x = keras.layers.LSTM(32, return_sequences=True)(input)
        x = keras.layers.LSTM(32, return_sequences=False)(x)
        
        outputs = {label: keras.layers.Dense(1, activation='sigmoid', name=label)(x) for label in output_labels}

        keras_model = keras.Model(inputs=input, outputs=outputs)

        keras_model.compile(
            optimizer=keras.optimizers.Adam(2e-4),
            loss={label: 'binary_crossentropy' for label in output_labels},
            metrics=[keras.metrics.BinaryAccuracy(threshold=0.5), keras.metrics.Precision(), keras.metrics.Recall()],
        )

        return keras_model
    
class ResNetLSTM(Model):
    def resblock(x, filter_sizes):
        res_conn = x

        res_conn = keras.layers.Conv2D(filter_sizes[2], kernel_size=(7,7), padding='same')(res_conn)
        res_conn = keras.layers.BatchNormalization()(res_conn)

        x = keras.layers.Conv2D(filter_sizes[0], kernel_size=(3,3), padding='same')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)

        x = keras.layers.Conv2D(filter_sizes[1], kernel_size=(3,3), padding='same')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)

        x = keras.layers.Conv2D(filter_sizes[2], kernel_size=(3,3), padding='same')(x)
        x = keras.layers.BatchNormalization()(x)

        x = keras.layers.Add()([x, res_conn])
        
        x = keras.layers.ReLU()(x)

        return x


    def create_model(output_labels: list, input_shape: tuple) -> keras.Model:
        input = keras.Input(shape=input_shape)
        x = keras.layers.Reshape((*input_shape, 1))(input)
        x = ResNetLSTM.resblock(x, (32, 64, 64))
        x = ResNetLSTM.resblock(x, (64, 128, 128))
        x = ResNetLSTM.resblock(x, (128, 128, 128))
        x = ResNetLSTM.resblock(x, (128, 64, 64))
        x = ResNetLSTM.resblock(x, (64, 64, 32))
        x = ResNetLSTM.resblock(x, (32, 16, 16))
        x = keras.layers.TimeDistributed(keras.layers.Flatten())(x)
        x = keras.layers.Bidirectional(keras.layers.LSTM(512, return_sequences=True))(x)
        x = keras.layers.Dropout(0.2)(x)
        x = keras.layers.Bidirectional(keras.layers.LSTM(512))(x)
        x = keras.layers.Dropout(0.4)(x)

        outputs = {label: keras.layers.Dense(1, activation='sigmoid', name=label)(x) for label in output_labels}

        keras_model = keras.Model(inputs=input, outputs=outputs)

        keras_model.compile(
            optimizer=keras.optimizers.Adam(1e-4),
            loss={label: 'binary_crossentropy' for label in output_labels},
            metrics=[keras.metrics.BinaryAccuracy(threshold=0.5)],
        )

        return keras_model

class ConvLSTM(Model):
    def create_model(output_labels: list, 
                    input_shape: tuple, 
                    num_timesteps=64, 
                    num_conv_layers=2, 
                    num_conv_filters=(32, 64), 
                    kern_size=(5,1),
                    num_lstm_layers=2,
                    hidden_layer_neurons=0) -> keras.Model:

        input = keras.Input(shape=input_shape)
        x = keras.layers.Reshape((*input_shape, 1))(input)
        x = keras.layers.Permute((2, 1, 3))(x)

        for i in range(num_conv_layers - 1):
            x = keras.layers.Conv2D(num_conv_filters[i], kernel_size=kern_size)(x)
            x = keras.layers.MaxPooling2D(pool_size=(2,2))(x)

        x = keras.layers.Conv2D(num_conv_filters[num_conv_layers-1], kernel_size=kern_size)(x)

        x = keras.layers.Reshape((-1, num_timesteps))(x)

        for _ in range(num_lstm_layers-1):
            x = keras.layers.LSTM(32, return_sequences=True)(x)

        x = keras.layers.LSTM(32, return_sequences=False)(x)

        if (hidden_layer_neurons > 0):
            x = keras.layers.Dense(hidden_layer_neurons, activation='relu')(x)

        outputs = {label: keras.layers.Dense(1, activation='sigmoid', name=label)(x) for label in output_labels}

        keras_model = keras.Model(inputs=input, outputs=outputs)

        keras_model.compile(
            optimizer=keras.optimizers.Adam(2e-4),
            loss={label: 'binary_crossentropy' for label in output_labels},
            metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()],
        )

        return keras_model
