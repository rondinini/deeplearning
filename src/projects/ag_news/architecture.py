import tensorflow as tf
from tensorflow.keras.layers import (
    Dense, Lambda, BatchNormalization, Dropout, 
    Activation, GRU, Embedding, Conv1D)
from tensorflow.keras import Model, Input
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K

from src.models.models import FunctionalModel

optimizer = Adam(learning_rate=0.001)

class CnnGRU(FunctionalModel):

    def __init__(self, **args):
        super(CnnGRU, self).__init__(**args)

    def connect_layers(self, vocab_size, vect_layer):
        input_layer = Input(shape=(None, ), dtype=tf.string)
        x = vect_layer(input_layer)
        mask = Lambda(lambda inputs: K.not_equal(inputs, '<pad>'))(input_layer)
        x = Embedding(vocab_size + 1, 128)(x)
        x = Conv1D(
            filters=32,
            kernel_size=5,
            strides=2,
            padding='valid',
        )(x)
        x = Activation('relu')(x)
        x = Dropout(0.2)(x)
        x = BatchNormalization()(x)
        x = GRU(64)(x, mask=mask)
        x = Dropout(0.2)(x)
        output = Dense(4, activation='softmax')(x)

        model = self._build_model(inputs=input_layer, outputs=output)
        model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        return model


class GRUNet(FunctionalModel):

    def __init__(self, **args):
        super(GRUNet, self).__init__(**args)

    def connect_layers(self, vocab_size, vect_layer):
        input_layer = Input(shape=(None, ), dtype=tf.string)
        x = vect_layer(input_layer)
        mask = Lambda(lambda inputs: K.not_equal(inputs, '<pad>'))(input_layer)
        x = Embedding(vocab_size + 1, 128)(x)
        x = GRU(64, return_sequences=True)(x, mask=mask)
        x = GRU(64)(x, mask=mask)
        x = Dropout(0.2)(x)
        output = Dense(4, activation='softmax')(x)

        model = self._build_model(inputs=input_layer, outputs=output)
        model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        return model
        