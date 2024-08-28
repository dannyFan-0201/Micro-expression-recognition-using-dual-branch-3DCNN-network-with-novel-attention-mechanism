import tensorflow as tf
from tensorflow.keras import layers
from keras.layers import LSTM

class GRUModel(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super(GRUModel, self).__init__(**kwargs)
        self.units = units
        self.state_size = units
        self.reset_gate = layers.Dense(units, activation='sigmoid')
        self.update_gate = layers.Dense(units, activation='sigmoid')
        self.candidate = layers.Dense(units, activation='tanh')

    def call(self, inputs, states):
        prev_output = states[0]
        reset = self.reset_gate(tf.concat([inputs, prev_output], axis=-1))
        update = self.update_gate(tf.concat([inputs, prev_output], axis=-1))
        candidate = self.candidate(tf.concat([inputs, reset * prev_output], axis=-1))
        output = (1 - update) * prev_output + update * candidate
        return output, [output]
class LSTMpeephole(LSTM):
    def __init__(self, units, peephole=True, **kwargs):
        self.P_i = None
        self.P_f = None
        self.P_c = None
        self.P_o = None
        super(LSTMpeephole, self).__init__(units, **kwargs)

    def build(self, input_shape):
        super(LSTMpeephole, self).build(input_shape)
        self.P_i = self.add_weight(shape=(self.units, self.units),
                                   initializer=self.kernel_initializer,
                                   name='P_i',
                                   trainable=True)
        self.P_f = self.add_weight(shape=(self.units, self.units),
                                   initializer=self.kernel_initializer,
                                   name='P_f',
                                   trainable=True)
        self.P_c = self.add_weight(shape=(self.units, self.units),
                                   initializer=self.kernel_initializer,
                                   name='P_c',
                                   trainable=True)
        self.P_o = self.add_weight(shape=(self.units, self.units),
                                   initializer=self.kernel_initializer,
                                   name='P_o',
                                   trainable=True)

    def step(self, x, states):
        assert len(states) == 2
        h_tm1 = states[0]
        c_tm1 = states[1]

        x_i = K.dot(x, self.W_i) + self.b_i
        x_f = K.dot(x, self.W_f) + self.b_f
        x_c = K.dot(x, self.W_c) + self.b_c
        x_o = K.dot(x, self.W_o) + self.b_o

        i = self.inner_activation(x_i + K.dot(h_tm1, self.U_i) + K.dot(c_tm1, self.P_i))
        f = self.inner_activation(x_f + K.dot(h_tm1, self.U_f) + K.dot(c_tm1, self.P_f))
        c = f * c_tm1 + i * self.activation(x_c + K.dot(h_tm1, self.U_c) + K.dot(c_tm1, self.P_c))
        o = self.inner_activation(x_o + K.dot(h_tm1, self.U_o) + K.dot(c_tm1, self.P_o))
        h = o * self.activation(c)
        return h, [h, c]