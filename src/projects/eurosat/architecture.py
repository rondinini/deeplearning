import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, BatchNormalization, Dropout, Activation, GlobalMaxPooling2D
from tensorflow.keras import Input
from tensorflow.keras.optimizers.legacy import Adam

from src.models.models import FunctionalModel
from src.models.layers import ResNetBlock

optimizer = Adam(learning_rate=0.001)

class FCNNetwork(FunctionalModel):

    def __init__(self, **args):
        super(FCNNetwork, self).__init__(**args)

    def connect_layers(self):
        input_layer = Input(shape=(None, None, 3))
        conv = Conv2D(
            filters=32, 
            kernel_size=5,
            strides=1,
            padding='same',
        )(input_layer)
        conv = BatchNormalization()(conv)
        conv = Activation('relu')(conv)
        conv = Dropout(0.3)(conv)
        conv = MaxPool2D(pool_size=2)(conv)

        conv = Conv2D(
            filters=64, 
            kernel_size=3, 
            strides=1,
            padding='same',
        )(conv)
        conv = BatchNormalization()(conv)
        conv = Activation('relu')(conv)
        conv = Dropout(0.3)(conv)
        conv = MaxPool2D(pool_size=2)(conv)

        conv = Conv2D(
            filters=256, 
            kernel_size=3, 
            padding='same',
        )(conv)
        conv = BatchNormalization()(conv)
        conv = Activation('relu')(conv)
        conv = Dropout(0.3)(conv)
        conv = MaxPool2D(pool_size=2)(conv)
        
        conv = Conv2D(
            filters=10, # number of output classes
            kernel_size=1,
            strides=1,
        )(conv)
        conv = BatchNormalization()(conv)
        conv = GlobalMaxPooling2D()(conv)
        output = Activation('softmax')(conv)

        model = self._build_model(inputs=input_layer, outputs=output)
        model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        return model

class CNNNetwork(FunctionalModel):

    def __init__(self, **args):
        super(CNNNetwork, self).__init__(**args)

    def connect_layers(self):
        input_layer = Input(shape=(64, 64, 3))
        conv = Conv2D(
            filters=32, 
            kernel_size=5,
            strides=1,
            padding='same',
        )(input_layer)
        conv = BatchNormalization()(conv)
        conv = Activation('relu')(conv)
        conv = Dropout(0.3)(conv)
        conv = MaxPool2D(pool_size=2)(conv)

        conv = Conv2D(
            filters=64, 
            kernel_size=3, 
            strides=1,
            padding='same',
        )(conv)
        conv = BatchNormalization()(conv)
        conv = Activation('relu')(conv)
        conv = Dropout(0.3)(conv)
        conv = MaxPool2D(pool_size=2)(conv)

        conv = Conv2D(
            filters=128, 
            kernel_size=3, 
            padding='same',
        )(conv)
        conv = BatchNormalization()(conv)
        conv = Activation('relu')(conv)
        conv = Dropout(0.3)(conv)
        conv = MaxPool2D(pool_size=2)(conv)
        
        conv = Flatten()(conv)

        dense_output = Dense(10)(conv)
        output = Activation('softmax')(dense_output)

        model = self._build_model(inputs=input_layer, outputs=output)
        model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        return model


class CnnResBlock(FunctionalModel):

    def __init__(self, **args):
        super(CnnResBlock, self).__init__(**args)
        self.resnet_one_params = {
            'filters': 32,
            'kernel_size': 3,
            'padding': 'same',
            'activation': 'relu',
            'dropout': 0.2,
            'use_max_pool': False,
            'strides': 2,
        }
        self.resnet_two_params = {
            'filters': 64,
            'kernel_size': 3,
            'padding': 'same',
            'activation': 'relu',
            'dropout': 0.2,
            'use_max_pool': False,
            'strides': 2,
        }
        self.resnet_three_params = {
            'filters': 128,
            'kernel_size': 3,
            'padding': 'same',
            'activation': 'relu',
            'dropout': 0.2,
            'use_max_pool': False,
            'strides': 1,
        }

    def connect_layers(self):
        input_layer = Input(shape=(64, 64, 3))
        resnet = ResNetBlock(**self.resnet_one_params)(input_layer)
        # resnet = MaxPool2D(pool_size=2)(resnet)
        resnet = ResNetBlock(**self.resnet_two_params)(resnet)
        # resnet = MaxPool2D(pool_size=2)(resnet)
        resnet = ResNetBlock(**self.resnet_three_params)(resnet)
        # resnet = MaxPool2D(pool_size=2)(resnet)

        resnet = Flatten()(resnet)

        dense_output = Dense(10)(resnet)
        output = Activation('softmax')(dense_output)

        model = self._build_model(inputs=input_layer, outputs=output)
        model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        return model