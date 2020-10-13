from tensorflow.keras import backend as backend
from tensorflow.keras.optimizers import Adam, RMSprop, SGD, Adagrad, Adadelta
from tensorflow.keras.layers import Flatten, Dense, Input, Dropout, GaussianNoise
from tensorflow.keras.layers import Conv1D, Conv2D, AveragePooling2D, MaxPooling1D, MaxPooling2D, BatchNormalization, AveragePooling1D
from keras_adabound import AdaBound
from tensorflow.keras.models import Sequential
import random


class NeuralNetwork:

    def __init__(self):
        self.activation_function = "relu"
        self.filters = 1
        self.kernel_size = 1
        self.stride = 1
        self.learning_rate = 1e-4
        self.neurons = 100
        self.conv_layers = 1
        self.dense_layers = 1
        self.optimizer = "RMSprop"

    def get_random_hyper_parameters(self):
        return {
            "activation_function": self.activation_function,
            "filters": self.filters,
            "stride": self.stride,
            "kernel_size": self.kernel_size,
            "learning_rate": self.learning_rate,
            "neurons": self.neurons,
            "conv_layers": self.conv_layers,
            "dense_layers": self.dense_layers,
            "optimizer": self.optimizer.__class__.__name__
        }

    def cnn_supervised(self, classes, number_of_samples):
        model = Sequential()
        model.add(Conv1D(filters=8, kernel_size=40, strides=4, activation='relu', padding='valid',
                         input_shape=(number_of_samples, 1)))
        model.add(Flatten())
        model.add(Dense(100, activation='relu', kernel_initializer='random_uniform', bias_initializer='zeros'))
        model.add(Dense(100, activation='relu', kernel_initializer='random_uniform', bias_initializer='zeros'))
        model.add(Dense(classes, activation='softmax'))
        model.summary()
        optimizer = RMSprop(lr=0.0001)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        return model

    def cnn_cswap_pointer(self, classes, number_of_samples):
        model = Sequential()
        model.add(Conv1D(filters=8, kernel_size=40, strides=4, activation='relu', padding='valid',
                         input_shape=(number_of_samples, 1)))
        model.add(Conv1D(filters=16, kernel_size=40, strides=4, activation='relu', padding='valid'))
        model.add(Conv1D(filters=32, kernel_size=40, strides=4, activation='relu', padding='valid'))
        model.add(Flatten())
        model.add(Dense(100, activation='relu', kernel_initializer='random_uniform', bias_initializer='zeros'))
        model.add(Dense(100, activation='relu', kernel_initializer='random_uniform', bias_initializer='zeros'))
        model.add(Dense(classes, activation='softmax'))
        model.summary()
        optimizer = RMSprop(lr=0.0001)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        return model

    def cnn_cswap_pointer_dropout(self, classes, number_of_samples):
        model = Sequential()
        model.add(Conv1D(filters=8, kernel_size=40, strides=4, activation='relu', padding='valid',
                         input_shape=(number_of_samples, 1)))
        model.add(Conv1D(filters=16, kernel_size=40, strides=4, activation='relu', padding='valid'))
        model.add(Conv1D(filters=32, kernel_size=40, strides=4, activation='relu', padding='valid'))
        model.add(Flatten())
        model.add(Dropout(0.5))
        model.add(Dense(100, activation='relu', kernel_initializer='random_uniform', bias_initializer='zeros'))
        model.add(Dropout(0.5))
        model.add(Dense(100, activation='relu', kernel_initializer='random_uniform', bias_initializer='zeros'))
        model.add(Dense(classes, activation='softmax'))
        model.summary()
        optimizer = RMSprop(lr=0.0001)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        return model

    def cnn_cswap_pointer_random(self, classes, number_of_samples):

        activation_function = ['relu', 'tanh', 'selu', 'elu'][random.randint(0, 3)]
        neurons = random.randrange(100, 400, 50)
        layers = random.randint(1, 5)
        kernel_size = random.randrange(10, 40, 5)
        stride = random.randrange(1, 4, 1)
        filters = random.randrange(4, 8, 1)

        model = Sequential()
        model.add(Conv1D(filters=filters, kernel_size=kernel_size, strides=stride, activation=activation_function, padding='valid',
                         input_shape=(number_of_samples, 1)))
        model.add(Conv1D(filters=filters * 2, kernel_size=kernel_size, strides=stride, activation=activation_function, padding='valid'))
        model.add(Conv1D(filters=filters * 4, kernel_size=kernel_size, strides=stride, activation=activation_function, padding='valid'))
        model.add(Flatten())
        for l_i in range(layers):
            model.add(Dense(neurons, activation=activation_function, kernel_initializer='random_uniform', bias_initializer='zeros'))
        model.add(Dense(classes, activation='softmax'))
        model.summary()
        optimizer = RMSprop(lr=0.0001)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        return model

    def cnn_cswap_pointer_dropout_random(self, classes, number_of_samples):
        activation_function = ['relu', 'tanh', 'selu', 'elu'][random.randint(0, 3)]
        neurons = random.randrange(100, 400, 50)
        layers = random.randint(1, 5)
        kernel_size = random.randrange(10, 40, 5)
        stride = random.randrange(1, 4, 1)
        filters = random.randrange(4, 8, 1)

        model = Sequential()
        model.add(Conv1D(filters=filters, kernel_size=kernel_size, strides=stride, activation=activation_function, padding='valid',
                         input_shape=(number_of_samples, 1)))
        model.add(Conv1D(filters=filters * 2, kernel_size=kernel_size, strides=stride, activation=activation_function, padding='valid'))
        model.add(Conv1D(filters=filters * 4, kernel_size=kernel_size, strides=stride, activation=activation_function, padding='valid'))
        model.add(Flatten())
        for l_i in range(layers):
            model.add(Dropout(0.5))
            model.add(Dense(neurons, activation=activation_function, kernel_initializer='random_uniform', bias_initializer='zeros'))
        model.add(Dense(classes, activation='softmax'))
        model.summary()
        optimizer = RMSprop(lr=0.0001)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        return model

    def cnn_cswap_arith(self, classes, number_of_samples):
        model = Sequential()
        model.add(AveragePooling1D(pool_size=4, strides=4, padding='valid', input_shape=(number_of_samples, 1)))
        model.add(BatchNormalization())
        model.add(Conv1D(filters=8, kernel_size=20, strides=1, activation='relu', padding='valid'))
        model.add(MaxPooling1D(pool_size=2, strides=2, padding='valid'))
        model.add(Conv1D(filters=16, kernel_size=20, strides=1, activation='relu', padding='valid'))
        model.add(Conv1D(filters=32, kernel_size=20, strides=1, activation='relu', padding='valid'))
        model.add(Flatten())
        model.add(Dense(100, activation='relu', kernel_initializer='he_uniform', bias_initializer='zeros'))
        model.add(Dense(100, activation='relu', kernel_initializer='he_uniform', bias_initializer='zeros'))
        model.add(Dense(classes, activation='softmax'))
        model.summary()
        optimizer = RMSprop(lr=0.0001)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        return model

    def cnn_cswap_arith_dropout(self, classes, number_of_samples):
        model = Sequential()
        model.add(AveragePooling1D(pool_size=4, strides=4, padding='valid', input_shape=(number_of_samples, 1)))
        model.add(BatchNormalization())
        model.add(Conv1D(filters=8, kernel_size=20, strides=1, activation='relu', padding='valid'))
        model.add(MaxPooling1D(pool_size=2, strides=2, padding='valid'))
        model.add(Conv1D(filters=16, kernel_size=20, strides=1, activation='relu', padding='valid'))
        model.add(Conv1D(filters=32, kernel_size=20, strides=1, activation='relu', padding='valid'))
        model.add(Flatten())
        model.add(Dropout(0.5))
        model.add(Dense(100, activation='relu', kernel_initializer='he_uniform', bias_initializer='zeros'))
        model.add(Dropout(0.5))
        model.add(Dense(100, activation='relu', kernel_initializer='he_uniform', bias_initializer='zeros'))
        model.add(Dense(classes, activation='softmax'))
        model.summary()
        optimizer = RMSprop(lr=0.0001)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        return model

    def cnn_cswap_arith_random(self, classes, number_of_samples):

        activation_function = ['relu', 'tanh', 'selu', 'elu'][random.randint(0, 3)]
        neurons = random.randrange(100, 400, 50)
        layers = random.randint(1, 5)
        kernel_size = random.randrange(10, 40, 5)
        stride = random.randrange(1, 4, 1)
        filters = random.randrange(4, 8, 1)

        model = Sequential()
        model.add(AveragePooling1D(pool_size=4, strides=4, padding='valid', input_shape=(number_of_samples, 1)))
        model.add(BatchNormalization())
        model.add(Conv1D(filters=filters, kernel_size=kernel_size, strides=stride, activation=activation_function, padding='valid'))
        model.add(Conv1D(filters=filters * 2, kernel_size=kernel_size, strides=stride, activation=activation_function, padding='valid'))
        model.add(Conv1D(filters=filters * 4, kernel_size=kernel_size, strides=stride, activation=activation_function, padding='valid'))
        model.add(Flatten())
        for l_i in range(layers):
            model.add(Dense(neurons, activation=activation_function, kernel_initializer='random_uniform', bias_initializer='zeros'))
        model.add(Dense(classes, activation='softmax'))
        model.summary()
        optimizer = RMSprop(lr=0.0001)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        return model

    def cnn_cswap_arith_dropout_random(self, classes, number_of_samples):
        activation_function = ['relu', 'tanh', 'selu', 'elu'][random.randint(0, 3)]
        neurons = random.randrange(100, 400, 50)
        layers = random.randint(1, 5)
        kernel_size = random.randrange(10, 40, 5)
        stride = random.randrange(1, 4, 1)
        filters = random.randrange(4, 8, 1)

        model = Sequential()
        model.add(AveragePooling1D(pool_size=4, strides=4, padding='valid', input_shape=(number_of_samples, 1)))
        model.add(BatchNormalization())
        model.add(Conv1D(filters=filters, kernel_size=kernel_size, strides=stride, activation=activation_function, padding='valid'))
        model.add(Conv1D(filters=filters * 2, kernel_size=kernel_size, strides=stride, activation=activation_function, padding='valid'))
        model.add(Conv1D(filters=filters * 4, kernel_size=kernel_size, strides=stride, activation=activation_function, padding='valid'))
        model.add(Flatten())
        for l_i in range(layers):
            model.add(Dropout(0.5))
            model.add(Dense(neurons, activation=activation_function, kernel_initializer='random_uniform', bias_initializer='zeros'))
        model.add(Dense(classes, activation='softmax'))
        model.summary()
        optimizer = RMSprop(lr=0.0001)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        return model

    # -- optimized GV for HA DL paper --- #

    def cnn_cswap_pointer_gv(self, classes, number_of_samples):
        model = Sequential()
        model.add(AveragePooling1D(pool_size=1, strides=1, padding='valid', input_shape=(number_of_samples, 1)))
        model.add(BatchNormalization())
        model.add(Conv1D(filters=8, kernel_size=10, strides=1, activation='relu', padding='valid'))
        model.add(Conv1D(filters=16, kernel_size=10, strides=1, activation='relu', padding='valid'))
        model.add(Conv1D(filters=32, kernel_size=10, strides=1, activation='relu', padding='valid'))
        model.add(Flatten())
        model.add(Dense(100, activation='relu', kernel_initializer='random_uniform', bias_initializer='zeros'))
        model.add(Dense(100, activation='relu', kernel_initializer='random_uniform', bias_initializer='zeros'))
        model.add(Dense(classes, activation='softmax'))
        model.summary()
        optimizer = RMSprop(lr=0.0001)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        return model

    def cnn_cswap_pointer_dropout_gv(self, classes, number_of_samples):
        model = Sequential()
        model.add(AveragePooling1D(pool_size=1, strides=1, padding='valid', input_shape=(number_of_samples, 1)))
        model.add(BatchNormalization())
        model.add(Conv1D(filters=8, kernel_size=10, strides=1, activation='relu', padding='valid'))
        model.add(Conv1D(filters=16, kernel_size=10, strides=1, activation='relu', padding='valid'))
        model.add(Conv1D(filters=32, kernel_size=10, strides=1, activation='relu', padding='valid'))
        model.add(Flatten())
        model.add(Dropout(0.5))
        model.add(Dense(100, activation='relu', kernel_initializer='random_uniform', bias_initializer='zeros'))
        model.add(Dropout(0.5))
        model.add(Dense(100, activation='relu', kernel_initializer='random_uniform', bias_initializer='zeros'))
        model.add(Dense(classes, activation='softmax'))
        model.summary()
        optimizer = RMSprop(lr=0.0001)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        return model

    def cnn_cswap_pointer_random_gv(self, classes, number_of_samples):

        activation_function = ['relu', 'tanh', 'selu', 'elu'][random.randint(0, 3)]
        neurons = random.randrange(100, 400, 50)
        layers = random.randint(1, 5)
        kernel_size = random.randrange(5, 10, 5)
        stride = random.randrange(1, 2, 1)
        filters = random.randrange(4, 8, 1)

        model = Sequential()
        model.add(AveragePooling1D(pool_size=1, strides=1, padding='valid', input_shape=(number_of_samples, 1)))
        model.add(BatchNormalization())
        model.add(Conv1D(filters=filters, kernel_size=kernel_size, strides=stride, activation=activation_function, padding='valid'))
        model.add(Conv1D(filters=filters * 2, kernel_size=kernel_size, strides=stride, activation=activation_function, padding='valid'))
        model.add(Conv1D(filters=filters * 4, kernel_size=kernel_size, strides=stride, activation=activation_function, padding='valid'))
        model.add(Flatten())
        for l_i in range(layers):
            model.add(Dense(neurons, activation=activation_function, kernel_initializer='random_uniform', bias_initializer='zeros'))
        model.add(Dense(classes, activation='softmax'))
        model.summary()
        optimizer = RMSprop(lr=0.0001)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        return model

    def cnn_cswap_pointer_dropout_random_gv(self, classes, number_of_samples):
        activation_function = ['relu', 'tanh', 'selu', 'elu'][random.randint(0, 3)]
        neurons = random.randrange(100, 400, 50)
        layers = random.randint(1, 5)
        kernel_size = random.randrange(5, 10, 5)
        stride = random.randrange(1, 2, 1)
        filters = random.randrange(4, 8, 1)

        model = Sequential()
        model.add(AveragePooling1D(pool_size=1, strides=1, padding='valid', input_shape=(number_of_samples, 1)))
        model.add(BatchNormalization())
        model.add(Conv1D(filters=filters, kernel_size=kernel_size, strides=stride, activation=activation_function, padding='valid'))
        model.add(Conv1D(filters=filters * 2, kernel_size=kernel_size, strides=stride, activation=activation_function, padding='valid'))
        model.add(Conv1D(filters=filters * 4, kernel_size=kernel_size, strides=stride, activation=activation_function, padding='valid'))
        model.add(Flatten())
        for l_i in range(layers):
            model.add(Dropout(0.5))
            model.add(Dense(neurons, activation=activation_function, kernel_initializer='random_uniform', bias_initializer='zeros'))
        model.add(Dense(classes, activation='softmax'))
        model.summary()
        optimizer = RMSprop(lr=0.0001)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        return model

    def cnn_cswap_arith_gv(self, classes, number_of_samples):
        model = Sequential()
        model.add(AveragePooling1D(pool_size=1, strides=1, padding='valid', input_shape=(number_of_samples, 1)))
        model.add(BatchNormalization())
        model.add(Conv1D(filters=8, kernel_size=20, strides=1, activation='relu', padding='valid'))
        model.add(MaxPooling1D(pool_size=2, strides=2, padding='valid'))
        model.add(Conv1D(filters=16, kernel_size=20, strides=1, activation='relu', padding='valid'))
        model.add(Conv1D(filters=32, kernel_size=20, strides=1, activation='relu', padding='valid'))
        model.add(Flatten())
        model.add(Dense(100, activation='relu', kernel_initializer='he_uniform', bias_initializer='zeros'))
        model.add(Dense(100, activation='relu', kernel_initializer='he_uniform', bias_initializer='zeros'))
        model.add(Dense(classes, activation='softmax'))
        model.summary()
        optimizer = RMSprop(lr=0.0001)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        return model

    def cnn_cswap_arith_dropout_gv(self, classes, number_of_samples):
        model = Sequential()
        model.add(AveragePooling1D(pool_size=1, strides=1, padding='valid', input_shape=(number_of_samples, 1)))
        model.add(BatchNormalization())
        model.add(Conv1D(filters=8, kernel_size=20, strides=1, activation='relu', padding='valid'))
        model.add(MaxPooling1D(pool_size=2, strides=2, padding='valid'))
        model.add(Conv1D(filters=16, kernel_size=20, strides=1, activation='relu', padding='valid'))
        model.add(Conv1D(filters=32, kernel_size=20, strides=1, activation='relu', padding='valid'))
        model.add(Flatten())
        model.add(Dropout(0.5))
        model.add(Dense(100, activation='relu', kernel_initializer='he_uniform', bias_initializer='zeros'))
        model.add(Dropout(0.5))
        model.add(Dense(100, activation='relu', kernel_initializer='he_uniform', bias_initializer='zeros'))
        model.add(Dense(classes, activation='softmax'))
        model.summary()
        optimizer = RMSprop(lr=0.0001)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        return model

    def cnn_cswap_arith_random_gv(self, classes, number_of_samples):

        activation_function = ['relu', 'tanh', 'selu', 'elu'][random.randint(0, 3)]
        neurons = random.randrange(100, 400, 50)
        layers = random.randint(1, 5)
        kernel_size = random.randrange(10, 20, 5)
        stride = random.randrange(1, 4, 1)
        filters = random.randrange(4, 8, 1)

        model = Sequential()
        model.add(AveragePooling1D(pool_size=1, strides=1, padding='valid', input_shape=(number_of_samples, 1)))
        model.add(BatchNormalization())
        model.add(Conv1D(filters=filters, kernel_size=kernel_size, strides=stride, activation=activation_function, padding='valid'))
        model.add(Conv1D(filters=filters * 2, kernel_size=kernel_size, strides=stride, activation=activation_function, padding='valid'))
        model.add(Conv1D(filters=filters * 4, kernel_size=kernel_size, strides=stride, activation=activation_function, padding='valid'))
        model.add(Flatten())
        for l_i in range(layers):
            model.add(Dense(neurons, activation=activation_function, kernel_initializer='random_uniform', bias_initializer='zeros'))
        model.add(Dense(classes, activation='softmax'))
        model.summary()
        optimizer = RMSprop(lr=0.0001)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        return model

    def cnn_cswap_arith_dropout_random_gv(self, classes, number_of_samples):
        activation_function = ['relu', 'tanh', 'selu', 'elu'][random.randint(0, 3)]
        neurons = random.randrange(100, 400, 50)
        layers = random.randint(1, 5)
        kernel_size = random.randrange(10, 20, 5)
        stride = random.randrange(1, 4, 1)
        filters = random.randrange(4, 8, 1)

        model = Sequential()
        model.add(AveragePooling1D(pool_size=1, strides=1, padding='valid', input_shape=(number_of_samples, 1)))
        model.add(BatchNormalization())
        model.add(Conv1D(filters=filters, kernel_size=kernel_size, strides=stride, activation=activation_function, padding='valid'))
        model.add(Conv1D(filters=filters * 2, kernel_size=kernel_size, strides=stride, activation=activation_function, padding='valid'))
        model.add(Conv1D(filters=filters * 4, kernel_size=kernel_size, strides=stride, activation=activation_function, padding='valid'))
        model.add(Flatten())
        for l_i in range(layers):
            model.add(Dropout(0.5))
            model.add(Dense(neurons, activation=activation_function, kernel_initializer='random_uniform', bias_initializer='zeros'))
        model.add(Dense(classes, activation='softmax'))
        model.summary()
        optimizer = RMSprop(lr=0.0001)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        return model
