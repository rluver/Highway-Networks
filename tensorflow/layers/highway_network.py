import tensorflow as tf
from tensorflow.keras.layers import Dense, Layer


class HighwayNetwork(Layer):
    def __init__(self, **kwargs):
        super(HighwayNetwork, self).__init__()
        self.units = kwargs.get('units')
        self.bias = kwargs.get('bias')
        self.bias = tf.keras.initializers.Constant(value=self.bias)
    
    def call(self, x):
        H = Dense(units=self.units, activation='relu', name='H')(x)
        T = Dense(units=self.units, bias_initializer=self.bias, activation='sigmoid', name='T')(x)

        output = H*T + x * (1.0-T)

        return output
