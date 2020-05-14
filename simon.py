from collections import defaultdict
from operator import xor

import numpy as np
import numpy.random as rd
from pyquil import Program, get_qc
from pyquil.api import QuantumComputer
from pyquil.gates import H, MEASURE
from pyquil.api import WavefunctionSimulator

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
    def __init__(self, qc, f, num_bits, max_iterations):
        self.__qc = qc
        self.__f = f
        self.num_bits = num_bits
        self.max_iterations = max_iterations
        self.qubits = list(range(2 * num_bits))
        self.computational_qubits = self.qubits[:num_bits]
        self.helper_qubits = self.qubits[num_bits:]
        self.__oracle = self.__create_unitary_matrix()
        self.__circuit = self.__create_quantum_circuit()
        self.__executable = self.__compile()
        self.equations = []
        self.candidates = []

    def __create_unitary_matrix(self):
        #Create list of all inputs
        inputs = [np.binary_repr(i, 2*self.num_bits) for i in range(0, 2**(2*self.num_bits))]

        #Create empty emptry matrix 
        matrix_u_f = np.zeros(shape=(2**(2 * self.num_bits), 2**(2 * self.num_bits)))

        # #Iteratively set relevant values to 1 in each row of permutation matrix
        for i in range(0, len(inputs)):
            el = inputs[i]
            x = el[:self.num_bits]
            y = self.__f(x)
            output = x + self.bitwise_xor(el[self.num_bits:], y)
            j = inputs.index(output)
            matrix_u_f[i][j] = 1

        return matrix_u_f


    def __create_quantum_circuit(self):
        circuit = Program()
        circuit.defgate("ORACLE", self.__oracle)
        circuit.inst([H(i) for i in self.computational_qubits])
        circuit.inst(tuple(["ORACLE"] + self.qubits))
        circuit.inst([H(i) for i in self.computational_qubits])
        return circuit

    def __compile(self):
        circuit = Program()
        simon_ro = circuit.declare('ro', 'BIT', len(self.qubits))
        circuit += self.__circuit
        circuit += [MEASURE(qubit, ro) for qubit, ro in zip(self.qubits, simon_ro)]
        executable = self.__qc.compile(circuit)
        return executable

    def run(self):
        for i in range(0, self.max_iterations):
            for i in range(0, self.num_bits - 1):
                sample = np.array(self.__qc.run(self.__executable)[0], dtype=int)
                self.equations.append(sample[:self.num_bits])
        return self.solve_lin_system()
                
    def solve_lin_system(self):
        self.candidates = []
        for i in range(0, 2**self.num_bits):
            eq = True
            s = np.array(list(np.binary_repr(i, self.num_bits))).astype(np.int8)
            for y_list in self.equations:
                y = np.array(y_list)
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

wf_sim = WavefunctionSimulator()

""" 
# Test Code - Uncomment block to use

n = 2
test_secret = np.binary_repr(3, n)
def func_no_secret(x):
    func_dict = create_1to1_dict(mask=np.binary_repr(1, n))
    return func_dict[x]

def func_secret(x):
    func_dict = create_2to2_dict(mask=np.binary_repr(1, n), secret=test_secret)
    return func_dict[x]

qc = get_qc('9q-square-qvm')
qc.compiler.client.timeout = 10000
solver = Simon(qc, func_secret, n, 8)
candidates = solver.run()
print(candidates)
 """
