import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import tensorflow as tf

"""
Implements the different output scaling techniques
"""

class LocalSkolikRescaling(tf.keras.layers.Layer):
    def __init__(self, input_dim):
        super(LocalSkolikRescaling, self).__init__()
        self.input_dim = input_dim
        self.w = tf.Variable(
            initial_value=tf.ones(shape=(1,input_dim)), dtype="float32",
            trainable=True, name="obs-weights")

    def call(self, inputs):
        return tf.math.multiply((inputs+1)/2, tf.repeat(self.w,repeats=tf.shape(inputs)[0],axis=0))
    
class LocalExpectationRescaling(tf.keras.layers.Layer):
    def __init__(self, input_dim):
        super(LocalExpectationRescaling, self).__init__()
        self.input_dim = input_dim
        self.w = tf.Variable(
            initial_value=tf.ones(shape=(1,input_dim)), dtype="float32",
            trainable=True, name="obs-weights")

    def call(self, inputs):
        return tf.math.multiply(inputs, tf.repeat(self.w,repeats=tf.shape(inputs)[0],axis=0))
    
class GlobalExpectationRescaling(tf.keras.layers.Layer):
    def __init__(self, input_dim):
        super(GlobalExpectationRescaling, self).__init__()
        self.input_dim = input_dim
        self.w = tf.Variable(
            initial_value=tf.ones(shape=(1,1)), dtype="float32",
            trainable=True, name="obs-weights")

    def call(self, inputs):
        return tf.math.multiply(inputs, self.w)

class GlobalSkolikRescaling(tf.keras.layers.Layer):
    def __init__(self, input_dim):
        super(GlobalSkolikRescaling, self).__init__()
        self.input_dim = input_dim
        self.w = tf.Variable(
            initial_value=tf.ones(shape=(1,1)), dtype="float32",
            trainable=True, name="obs-weights")

    def call(self, inputs):
        return tf.math.multiply((inputs+1)/2, self.w)