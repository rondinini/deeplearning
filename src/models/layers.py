from tensorflow.keras.layers import Layer, Conv2D, BatchNormalization, Dropout, Activation, MaxPool2D


class CNNBlock(Layer):

    def __init__(self, **kwargs):
        super(CNNBlock, self).__init__()
        self.conv = Conv2D(
            filters=kwargs['filters'],
            kernel_size=kwargs['kernel_size'],
            padding=kwargs['padding'],
        )
        self.bn = BatchNormalization()
        self.activation = Activation(kwargs['activation'])
        self.do = Dropout(kwargs['dropout'])
        if kwargs['use_max_pool'] == True:
            self.max_pool = MaxPool2D(pool_size=kwargs['pool_size'])
        else:
            self.max_pool = False

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x)
        x = self.activation(x)
        x = self.do(x)
        if self.max_pool:
            x = self.max_pool(x)
        return x