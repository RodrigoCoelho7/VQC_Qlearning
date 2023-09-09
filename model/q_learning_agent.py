import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import tensorflow as tf
from vqc.data_reup_model import FullEncodingMultiQubitUniversalQuantumClassifier, BaselinePQC, DataReupPQC, UniversalQuantumClassifier, MultiQubitUniversalQuantumClassifier
from vqc.vqc_circuits import UQC, SkolikBaseline, LockwoodBaseline

"""
Initializes either the Quantum Q-Learning agent or the Neural Network Q-Learning agent.
"""

def QuantumQLearningAgent(vqc, quantum_model, observables, target,state_dim, rescaling_type, activation):
    input_tensor = tf.keras.Input(shape=(state_dim, ), dtype=tf.dtypes.float32, name='input')

    pqc = quantum_model(vqc,state_dim, observables, activation=activation)([input_tensor])
#    if isinstance(vqc, UQC):
#        pqc = MultiQubitUniversalQuantumClassifier(vqc,state_dim, observables, activation=activation)([input_tensor])
#    elif isinstance(vqc, SkolikBaseline) or isinstance(vqc, LockwoodBaseline):
#        pqc = BaselinePQC(vqc, state_dim, observables, activation=activation)([input_tensor])
#    else:
#        pqc = DataReupPQC(vqc,state_dim, observables, activation=activation)([input_tensor])

    process = tf.keras.Sequential([rescaling_type(len(observables))], name=target*"Target"+"Q-values")
    Q_values = process(pqc)
    model = tf.keras.Model(inputs=[input_tensor], outputs=Q_values)
    return model

def NNQLearningAgent(state_dim, num_actions, activation):
    input_tensor = tf.keras.Input(shape=(state_dim, ), dtype=tf.dtypes.float32, name='input')

    Q_values = tf.keras.Sequential([
        tf.keras.layers.Dense(state_dim, activation=activation, input_shape=(state_dim,), kernel_initializer='he_normal'),
        tf.keras.layers.Dense(10, activation=activation, kernel_initializer='he_normal'),
        tf.keras.layers.Dense(10, activation=activation, kernel_initializer='he_normal'),
        tf.keras.layers.Dense(num_actions, activation = 'linear', kernel_initializer='he_normal'),
    ])([input_tensor])

    model = tf.keras.Model(inputs=[input_tensor], outputs=Q_values)
    return model