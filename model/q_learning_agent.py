from model.output_scaling import LocalSkolikRescaling, LocalExpectationRescaling, GlobalExpectationRescaling, GlobalSkolikRescaling
from vqc.data_reup_model import MyPQC, UniversalQuantumClassifier
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import tensorflow as tf

def QLearningAgent(num_qubits, num_layers, observables, circuit_arch, data_reuploading, target, measurement,state_dim, rescaling_type = "localskolik"):
    input_tensor = tf.keras.Input(shape=(state_dim, ), dtype=tf.dtypes.float32, name='input')
    if circuit_arch == "uqc":
        pqc = UniversalQuantumClassifier(num_qubits, num_layers, observables, circuit_arch, data_reuploading, measurement, activation='tanh')([input_tensor])
    else:
        pqc = MyPQC(num_qubits, num_layers, observables, circuit_arch, data_reuploading, measurement, activation='tanh')([input_tensor])
    if rescaling_type == "localskolik":
        process = tf.keras.Sequential([LocalSkolikRescaling(len(observables))], name=target*"Target"+"Q-values")
    elif rescaling_type == "localexpectation":
        process = tf.keras.Sequential([LocalExpectationRescaling(len(observables))], name=target*"Target"+"Q-values")
    elif rescaling_type == "globalskolik":
        process = tf.keras.Sequential([GlobalSkolikRescaling(len(observables))], name=target*"Target"+"Q-values")
    elif rescaling_type == "globalexpectation":
        process = tf.keras.Sequential([GlobalExpectationRescaling(len(observables))], name=target*"Target"+"Q-values")
    elif rescaling_type == "none":
        process = tf.keras.Sequential([tf.keras.layers.Lambda(lambda x: x)], name=target*"Target"+"Q-values")
    else:
        raise ValueError("Rescaling type not recognized")
    Q_values = process(pqc)
    model = tf.keras.Model(inputs=[input_tensor], outputs=Q_values)
    return model

