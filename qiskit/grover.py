from qiskit import QuantumRegister, QuantumCircuit, execute, Aer
from qiskit.quantum_info.operators import Operator
from qiskit.visualization import plot_histogram

import numpy as np
import time
import random


def timer(original_function):
    '''
    A decorator function to time a function.
    '''
    def wrapper_function(*args,**kwargs):
        start=time.time()
        result=original_function(*args,**kwargs)
        stop=time.time()
        diff=stop-start
        print('{} took {} seconds\n'.format(original_function.__name__,diff))
        return result
    return wrapper_function


class Solver(object):

    def __init__(self, f, n, n_trials=10):
        '''
        Initialize the class

        Input: function, number of qubits, number of trials

        Additionally, the number of times G is to be run is evaluated.
        '''
        self.f = f
        self.n = n
        self.k = int(np.floor((np.pi/4)*np.sqrt(2**n)))

        self.n_trials = n_trials

        self.__build_circuit()

    def __generate_bit_strings(self, n):
        '''
        Input: n
        Output: A list of bit strings Ex. n=2 -> ["00", "01", "10", "11"]

        A recursive function to generate all possible bit strings with size n.
        '''
        if n==1:
            return ["0", "1"]
        else:
            return ["0"+x for x in self.__generate_bit_strings(n-1)]+["1"+x for x in self.__generate_bit_strings(n-1)]


    def __produce_z_0_gate(self):
        '''
        Produce matrix and gate for Z_0
        '''
        z_0 = np.identity(2**n)
        z_0[0][0] = -z_0[0][0]
        self.__z_0 = Operator(z_0)


    def __produce_z_f_gate(self):
        '''
        Produce matrix and gate for Z_f
        using the mapping between the input and output.
        '''
        z_f = np.identity(2**n)
        bit_strings = self.__generate_bit_strings(self.n)
        for bit_string in bit_strings:
            output = f(bit_string)
            if output == 1:
                i = bit_strings.index(bit_string)
        #i = np.random.randint(2**n)
        z_f[i][i] = -z_f[i][i]
        self.__z_f = Operator(z_f)


    def __produce_negative_gate(self):
        '''
        Produce matrix and gate for changing
        the coefficient of the set of qubits.
        '''
        negative =  -np.identity(2**n)
        self.__negative = Operator(negative)

    def __build_circuit(self):
        '''
        Build the circuit for Grover's algorithm
        '''
        self.__produce_z_f_gate()
        self.__produce_z_0_gate()
        self.__produce_negative_gate()

        G = QuantumCircuit(self.n, self.n)


        #The part of the Grover's algorithm circuit
        #which might repeated to obtain the correct solution.
        G.unitary(self.__z_f, [i for i in range(self.n)], label='z_f')
        for i in range(self.n):
            G.h(i)
        G.unitary(self.__z_0, [i for i in range(self.n)], label='z_0')
        for i in range(self.n):
            G.h(i)
        G.unitary(self.__negative, [i for i in range(self.n)], label='negative')
        
        #The main circuit for the algorithm
        self.__circuit = QuantumCircuit(self.n, self.n)
        for i in range(self.n):
            self.__circuit.h(i)
        for i in range(self.k):
            self.__circuit+=G
        self.__circuit.measure([i for i in range(self.n)],[i for i in range(self.n)])
    @timer
    def solve(self):
        '''
        Run and measure the quantum circuit
        and return the result.
        The circuit is run for n_trials number of trials.
        '''
        simulator = Aer.get_backend("qasm_simulator")
        job = execute(self.__circuit, simulator, shots=1000)

        # Grab results from the job
        result = job.result()

        # Returns counts
        counts = result.get_counts(self.__circuit)
        print("\nTotal count for 00 and 11 are:",counts)

        return counts

def random_bit_string_generator(n=1):
    '''
    Generates a random bit string of length n

    Input: n (Default: n=1)
    Output: A bit string Ex. n=7 -> "0101010"
    '''
    bit_string = ''
    for i in range(0,n):
        bit_string+=str(random.choice([0,1]))
    return bit_string

n=3
bit_string = random_bit_string_generator(n)
#Test function
f = lambda x: 1 if x==bit_string else 0

solver = Solver(f, n)
counts = solver.solve()

res = max(counts, key=counts.get)
print("Bit string for which f is 1: {}".format(res))