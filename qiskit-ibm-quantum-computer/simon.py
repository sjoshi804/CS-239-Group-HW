from collections import defaultdict
from operator import xor
import numpy as np
from qiskit import(
  QuantumCircuit,
  execute,
  Aer)
from qiskit import IBMQ

#Helper Functions
def bitwise_xor(a, b):
    if len(a) != len(b):
        raise ValueError("Arguments must have same length!")
    return "{0:0{1:0d}b}".format(xor(int(a, 2), int(b, 2)), len(a))

def create_1to1_dict(mask):
    num_bits = len(mask)
    func_as_dict = {}
    for x in range(2**num_bits):
        bit_vector = np.binary_repr(x, num_bits)
        func_as_dict[bit_vector] = bitwise_xor(bit_vector, mask)
    return func_as_dict

def create_2to2_dict(mask, secret):
    num_bits = len(mask)
    func_as_dict = {}
    for x in range(2**num_bits):
        bit_vector_1 = np.binary_repr(x, num_bits)
        if bit_vector_1 in func_as_dict:
            continue
        bit_vector_2 = bitwise_xor(bit_vector_1, secret)
        func_as_dict[bit_vector_1] = bitwise_xor(bit_vector_1, mask)
        func_as_dict[bit_vector_2] = func_as_dict[bit_vector_1]
    return func_as_dict

class Simon:
    #Constructor
    #f is a function that takes as input a string in binary and returns as output a string in binary
    def __init__(self, simulator, f, num_bits, max_iterations):
        self.__simulator = simulator
        self.__f = f
        self.num_bits = num_bits
        self.max_iterations = max_iterations
        self.qubits = list(range(2 * num_bits))
        self.computational_qubits = self.qubits[:num_bits]
        self.helper_qubits = self.qubits[-num_bits:]
        self.__oracle = self.__create_unitary_matrix()
        self.__circuit = self.__create_quantum_circuit()
        self.equations = []
        self.candidates = []
        self.timing_of_latest_run = None

    def __create_unitary_matrix(self):
        #Create list of all inputs
        inputs = [np.binary_repr(i, 2*self.num_bits) for i in range(0, 2**(2*self.num_bits))]

        #Create empty emptry matrix 
        matrix_u_f = np.zeros(shape=(2**(2 * self.num_bits), 2**(2 * self.num_bits)))

        # #Iteratively set relevant values to 1 in each row of permutation matrix
        for i in range(0, len(inputs)):
            el = inputs[i] 
            x = el[::-1][:self.num_bits] 
            y = self.__f(x) 
            output = (x + bitwise_xor(el[::-1][-self.num_bits:], y))[::-1] 
            j = inputs.index(output)
            matrix_u_f[i][j] = 1

        return matrix_u_f


    def __create_quantum_circuit(self):
        circuit = QuantumCircuit(self.num_bits * 2, self.num_bits)
        for i in self.computational_qubits:
            circuit.h(i)
        circuit.unitary(self.__oracle, self.qubits, label="Oracle")
        for i in self.computational_qubits:
            circuit.h(i)
        circuit.measure(self.computational_qubits, self.computational_qubits)
        return circuit

    def run(self):
        for i in range(0, self.max_iterations):
            job = execute(self.__circuit, self.__simulator, shots=self.num_bits-1)
            result = job.result().get_counts(self.__circuit)
            self.timing_of_latest_run = job.result().time_taken
            self.equations += list(result.keys())
        self.equations = list(dict.fromkeys(self.equations))
        return self.solve_lin_system()
                
    def solve_lin_system(self):
        self.candidates = []
        for i in range(0, 2**self.num_bits):
            eq = True
            s = np.array(list(np.binary_repr(i, self.num_bits))).astype(np.int8)
            for y_list in self.equations:
                y = np.array(list(y_list)[::-1]).astype(np.int8)
                if ((np.dot(s, y) % 2) != 0):
                    eq = False
                    break
            if eq:
                self.candidates.append(np.binary_repr(i, self.num_bits))
        return self.candidates

    def bitwise_xor(self, a, b):
        if len(a) != len(b):
            raise ValueError("Arguments must have same length!")
        return "{0:0{1:0d}b}".format(xor(int(a, 2), int(b, 2)), len(a))

    def get_time_of_latest_run(self):
        return self.timing_of_latest_run


# Test Code - Uncomment block to use
n = 2
test_secret = np.binary_repr(1, n)
def func_no_secret(x):
    func_dict = create_1to1_dict(mask=np.binary_repr(1, n))
    return func_dict[x]

def func_secret(x):
    func_dict = create_2to2_dict(mask=np.binary_repr(1, n), secret=test_secret)
    return func_dict[x]

# Using an actual quantum computer
API_CREDENTIAL = "Your API Key"
IBMQ.save_account(API_CREDENTIAL, overwrite=True)
IBMQ.load_account()
provider = IBMQ.get_provider(hub='ibm-q')
qc = provider.get_backend("ibmq_16_melbourne")
solver = Simon(qc, func_secret, n, 8)
candidates = solver.run()
print(candidates) 
print(solver.get_time_of_latest_run())
