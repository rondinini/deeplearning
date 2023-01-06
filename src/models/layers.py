from tensorflow.keras.layers import Layer, Conv2D, BatchNormalization, Dropout, Activation, MaxPool2D, Add


class CNNBlock(Layer):

    def __init__(self, **kwargs):
        super(CNNBlock, self).__init__()
        self.conv = Conv2D(
            filters=kwargs['filters'],
            kernel_size=kwargs['kernel_size'],
            padding=kwargs['padding'],
            strides=kwargs['strides'],
        )
        self.bn = BatchNormalization()
        self.activation = Activation(kwargs['activation'])
        self.do = Dropout(kwargs['dropout'])

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x)
        x = self.activation(x)
        x = self.do(x)
        return x


class ResNetBlock(Layer):

    def __init__(self, **kwargs):
        super(ResNetBlock, self).__init__()
        # Strides need to match for first and last cnn layer
        self.cnn_block_one = CNNBlock(**kwargs)
        self.cnn_block_skip = CNNBlock(**kwargs)
        # Strides can't change in between the ResNet
        kwargs['strides'] = 1
        self.cnn_block_two = CNNBlock(**kwargs)
        self.act_out = Activation('relu')
        self.bn_out = BatchNormalization()

    def call(self, inputs):
        x = self.cnn_block_one(inputs)
        x = self.cnn_block_two(x)

        skipped_x = self.cnn_block_skip(inputs)

        output = Add()([x, skipped_x])
        output = self.bn_out(output)
        output = self.act_out(output)

        return output


