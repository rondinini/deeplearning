import tensorflow as tf
from tensorflow.keras.layers import (Dense, Conv2D, MaxPool2D, Flatten, BatchNormalization, 
Dropout, Activation, GlobalMaxPooling2D, Reshape, Conv2DTranspose, LeakyReLU)
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
            filters=128, 
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


class GANGenerator(FunctionalModel):

    def __init__(self, **args):
        super(GANGenerator, self).__init__(**args)

    def connect_layers(self, codings_size):
        input_layer = Input(shape=(codings_size, ))
        x = Dense(16 * 16 * 128)(input_layer)
        x = Reshape((16, 16, 128))(x)
        x = BatchNormalization()(x)

        x = Conv2DTranspose(
            filters=64, 
            kernel_size=5, 
            strides=2, 
            padding='same'
        )(x)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)
        
        x = Conv2DTranspose(
            filters=3,
            kernel_size=5,
            strides=2,
            padding='same',
        )(x)
        output = Activation('tanh')(x)
        model = self._build_model(inputs=input_layer, outputs=output)
        return model


class GANDiscriminator(FunctionalModel):

    def __init__(self, **args):
        super(GANDiscriminator, self).__init__(**args)

    def connect_layers(self, batch_input_shape: tuple):
        input_layer = Input(batch_input_shape=batch_input_shape)
        x = Conv2D(
            filters=64,
            kernel_size=5,
            strides=2,
            padding='same',
        )(input_layer)
        x = LeakyReLU(0.2)(x)
        x = Dropout(0.4)(x)
        
        x = Conv2D(
            filters=128,
            kernel_size=5,
            strides=2,
            padding='same',
        )(x)
        x = LeakyReLU(0.2)(x)
        x = Dropout(0.4)(x)
        x = Flatten()(x)

        output = Dense(1, activation='sigmoid')(x)

        model = self._build_model(inputs=input_layer, outputs=output)
        model.compile(optimizer='rmsprop', loss='binary_crossentropy')
        return model


class GAN(FunctionalModel):

    def __init__(self, generator, discriminator, **args):
        super(GAN, self).__init__(**args)
        self.generator = generator
        self.discriminator = discriminator

    def connect_layers(self, codings_size):
        input_layer = Input(shape=(codings_size, ))
    
        generator_output = self.generator(input_layer)
        # Making the discriminator non trainable
        self.discriminator.trainable = False
        discriminator_output = self.discriminator(generator_output)

        model = self._build_model(inputs=input_layer, outputs=discriminator_output)
        model.compile(loss='binary_crossentropy', optimizer='rmsprop')
        return model