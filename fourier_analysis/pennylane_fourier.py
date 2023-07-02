
import pennylane as qml
from fourier_analysis.pennylane_operations import QMLOperations
from pennylane.fourier import coefficients, circuit_spectrum
from functools import partial
from pennylane.fourier.visualize import *
import matplotlib.pyplot as plt
import numpy as np

class Fourier_Analysis():
    """
    This class is used to create a pennylane quantum circuit and perform fourier analysis on it.
    The class takes the following arguments:
    num_qubits: The number of qubits in the circuit

    num_layers: The number of layers in the circuit

    circuit_arch: The architecture of the circuit. Currently only "skolik" is supported, but in the future
    more architectures will be added (lock, for example).

    data_reuploading: The type of data reuploading used in the circuit. (basic, schuld or baseline)

    measurement: The type of measurement used in the circuit. (Right now it doesn't really matter)

    weights: The weights of the circuit. The weight's shape depends on the circuit architecture
    and the data reuploading method:
        skolik + basic: (num_layers, num_qubits*2)
        skolik + schuld: (num_layers+1, num_qubits*2)
        skolik + baseline: (num_layers, num_qubits*2)

    data: The data used in the circuit. The data can be a single parameter or a list of parameters

    """

    def __init__(self, num_qubits, num_layers, circuit_arch, data_reuploading, measurement, weights, data, input_weights):
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.circuit_arch = circuit_arch
        self.data_reuploading = data_reuploading
        self.measurement = measurement
        self.weights = weights
        self.input_weights = input_weights
        self.data = data
        if type(data) == int or type(data) == float:
            self.data_dim = 1
        else:
            self.data_dim = len(data)
        self.operations = QMLOperations()
        self.circuit = self.create_circuit()

    
    #This function creates the circuit based on the arguments passed to the class.

    def create_circuit(self):
        dev = qml.device("default.qubit", wires = self.num_qubits)
        if self.circuit_arch == "skolik":
            if self.data_reuploading == "schuld":
                circuit = qml.QNode(self.operations.schuld_datareup, dev)
            elif self.data_reuploading == "basic":
                circuit = qml.QNode(self.operations.basic_datareup, dev)
            elif self.data_reuploading == "baseline":
                circuit = qml.QNode(self.operations.baseline_datareup, dev)
        elif self.circuit_arch == "uqc":
            circuit = qml.QNode(self.operations.uqc, dev)
        return circuit
    
    # This function draws the circuit
    
    def draw_circuit(self):
        if self.circuit_arch != "uqc":
            return print(qml.draw(self.circuit)(self.weights, self.num_layers, self.data_dim, self.data))
        else:
            return print(qml.draw(self.circuit)(self.weights, self.num_qubits, self.num_layers, self.data))
    
    # This function returns the circuit's spectrum
   # def circuit_spectrum(self):
   #     freqs =  circuit_spectrum(self.circuit)(self.weights, self.num_layers,self.data_dim, self.data)
   #     for i, freq in enumerate(list(freqs.keys())):
            

    
    """
    This function returns the circuit's fourier coefficients. If it receives a set of weights, it will
    return the fourier coefficients of the circuit with those weights. If it doesn't receive any weights,
    it will return the fourier coefficients of the circuit with the weights passed to the class.
    """
    
    def fourier_coefficients(self, weights = None):
        if weights is None:
            partial_circuit = partial(self.circuit, self.weights, self.num_layers, self.data_dim)
        else:
            partial_circuit = partial(self.circuit, weights, self.num_layers, self.data_dim)
        return coefficients(partial_circuit, self.data_dim, int(np.floor(len(self.circuit_spectrum()["x_0"])/2)))
    

    """
    This function plots the circuit's fourier coefficients. If it receives a set of weights, it will
    plot the fourier coefficients of the circuit with those weights. If it doesn't receive any weights,
    it will plot the fourier coefficients of the circuit with the weights passed to the class.
    """
    def plot_coefficients(self, weights = None):
        fig, ax = plt.subplots(2, 1, sharex=True, sharey=True) # Set up the axes
        bar(self.fourier_coefficients(weights), self.data_dim, ax)

    """
    This function plots the distribution of the circuit's fourier coefficients. It takes the following arguments:
    dist_type: The type of distribution to be plotted. It can be "violin", "box" or "panel".
    If it is "violin", it will plot a violin plot of the distribution.
    If it is "box", it will plot a box plot of the distribution.
    If it is "panel", it will plot a panel plot of the distribution.
    """

    def plot_random_coefficients_distribution(self, dist_type = "violin"):
        coeffs = []

        for _ in range(100):
            weights = np.random.uniform(0, 2*np.pi, size=(self.num_layers+1, self.num_qubits*2))
            c = coefficients(partial(self.circuit, weights, self.num_layers, self.data_dim), self.data_dim, self.num_layers)
            coeffs.append(np.round(c, decimals=8))

        if dist_type == "violin":
            fig, ax = plt.subplots(2, 1, sharey=True, figsize=(15, 4))
            violin(coeffs, self.data_dim, ax, show_freqs=True)

        elif dist_type == "box":
            # The subplot axes must be *polar* for the radial plots
            fig, ax = plt.subplots(
                1, 2, sharex=True, sharey=True,
                subplot_kw=dict(polar=True),
                figsize=(15, 8)
            )
            radial_box(coeffs, 2, ax, show_freqs=True, show_fliers=False)

        elif dist_type == "panel":
            assert(self.data_dim <= 2)
            # Need a grid large enough to hold all coefficients up to frequency 2
            fig, ax = plt.subplots(5, 5, figsize=(12, 10), sharex=True, sharey=True)
            panel(coeffs, 2, ax)



