import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import tensorflow as tf

def QLearningAgent(vqc, observables, target,state_dim, rescaling_type, activation, pqc):
    input_tensor = tf.keras.Input(shape=(state_dim, ), dtype=tf.dtypes.float32, name='input')

    quantum_model = pqc(vqc,state_dim, observables, activation=activation)([input_tensor])

    process = tf.keras.Sequential([rescaling_type(len(observables))], name=target*"Target"+"Q-values")
    Q_values = process(quantum_model)
    model = tf.keras.Model(inputs=[input_tensor], outputs=Q_values)
    return model