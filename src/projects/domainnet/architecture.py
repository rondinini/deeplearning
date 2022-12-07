import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, BatchNormalization, Dropout, Activation, GlobalMaxPooling2D
from tensorflow.keras import Input
from tensorflow.keras.optimizers import Adam

from src.models.models import FunctionalModel

optimizer = Adam(learning_rate=0.001)

class FCNDomainNet(FunctionalModel):

    def __init__(self, **args):
        super(FCNDomainNet, self).__init__(**args)

    def connect_layers(self):
        input_layer = Input(shape=(None, None, 3))
        conv = Conv2D(
            filters=64, 
            kernel_size=5,
            strides=1,
            padding='same',
        )(input_layer)
        conv = BatchNormalization()(conv)
        conv = Activation('relu')(conv)
        conv = Dropout(0.3)(conv)
        conv = MaxPool2D(pool_size=2)(conv)

        conv = Conv2D(
            filters=128, 
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
            filters=345, # number of output classes
            kernel_size=1,
            strides=1,
        )(conv)
        conv = BatchNormalization()(conv)
        conv = GlobalMaxPooling2D()(conv)
        output = Activation('softmax')(conv)

        model = self._build_model(inputs=input_layer, outputs=output)
        model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        return model
