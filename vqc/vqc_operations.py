import cirq


class OPERATIONS():

    def skolik_encoding(self,input,qubit):
        return cirq.rx(input).on(qubit)
    
    def lock_encoding(self,input,qubit):
        return [cirq.rx(input).on(qubit),
                cirq.rz(input).on(qubit)]
    
    def one_qubit_rotation(self,qubit, symbols):
        """
        Returns Cirq gates that apply a rotation of the bloch sphere about the X,
        Y and Z axis, specified by the values in 'symbols'
        """

        return [
            cirq.rx(symbols[0])(qubit),
            cirq.ry(symbols[1])(qubit),
            cirq.rz(symbols[2])(qubit)
        ]
    
    def parametrized_gates_skolik(self,qubit, symbols):
        return [
            cirq.ry(symbols[0])(qubit),
            cirq.rz(symbols[1])(qubit)
        ]

    def entangling_layer_skolik(self,qubits):
        """
        Returns a layer of CZ entangling gates on 'qubits' (arranged in a circular topology)
        """
        cz_ops = [cirq.CZ(q0,q1) for q0,q1 in zip(qubits, qubits[1:])]
        cz_ops += ([cirq.CZ(qubits[0], qubits[-1])] if len(qubits) != 2 else [])
        return cz_ops
    
    def parametrized_gates_lock(self,qubit,symbols):
      return self.one_qubit_rotation(qubit, symbols)

    def entangling_layer_lock(self,qubits):
      cnot_ops = [cirq.CNOT(control = self.qubits[i], target = self.qubits[i-1]) for i in range(1,len(self.qubits))]
      return cnot_ops
    
    def cnot_chain(self,qubits):
        cnot_ops = [cirq.CNOT(control = qubits[i], target = qubits[i+1]) for i in range(len(qubits)-1)]
        return cnot_ops
    
    def uqc(self,qubit, theta, input):
        return [
            cirq.rz(input)(qubit),
            cirq.ry(theta)(qubit),
        ]
