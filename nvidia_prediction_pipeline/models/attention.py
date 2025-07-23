# models/attention.py

import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense

class BahdanauAttention(Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = Dense(units)        
        self.V = Dense(1)

    def call(self, hidden_states):
        """
        hidden_states: Tensor of shape (batch, timesteps, features)
        Returns: context vector of shape (batch, features)
        """
        # score shape: (batch, timesteps, units)
        score = tf.nn.tanh(self.W1(hidden_states))

        # attention_weights shape: (batch, timesteps, 1)
        attention_weights = tf.nn.softmax(self.V(score), axis=1)

        # context_vector shape after sum: (batch, features)
        context_vector = attention_weights * hidden_states
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector