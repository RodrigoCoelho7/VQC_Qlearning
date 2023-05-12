from vqc.data_reup_model import DataReupPQC,BaselinePQC, UniversalQuantumClassifier, MultiQubitUniversalQuantumClassifier, FullEncodingMultiQubitUniversalQuantumClassifier
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import tensorflow as tf
from vqc.vqc_circuits import UQC, SkolikBaseline, LockwoodBaseline

def QLearningAgent(vqc, observables, target,state_dim, rescaling_type, activation):
    input_tensor = tf.keras.Input(shape=(state_dim, ), dtype=tf.dtypes.float32, name='input')

    if isinstance(vqc, UQC):
        pqc = UniversalQuantumClassifier(vqc,state_dim, observables, activation=activation)([input_tensor])
    elif isinstance(vqc, SkolikBaseline) or isinstance(vqc, LockwoodBaseline):
        pqc = BaselinePQC(vqc, observables, activation=activation)([input_tensor])
    else:
        pqc = DataReupPQC(vqc, observables, activation=activation)([input_tensor])

    process = tf.keras.Sequential([rescaling_type(len(observables))], name=target*"Target"+"Q-values")
    Q_values = process(pqc)
    model = tf.keras.Model(inputs=[input_tensor], outputs=Q_values)
    return model