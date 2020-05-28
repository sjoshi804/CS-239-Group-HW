from qiskit import QuantumRegister, QuantumCircuit, execute, Aer
from qiskit.quantum_info.operators import Operator
from qiskit.visualization import plot_histogram

import numpy as np
import time
import random


class Solver(object):

    def __init__(self, f, n):   
        self.f = f
        self.n = n

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

    def __get_tensor(self, bit_string):
        if bit_string=="0":
            return np.array([1,0], dtype=np.int16)
        elif bit_string=='1':
            return np.array([0,1], dtype=np.int16)
        else:
            return np.tensordot(self.__get_tensor(bit_string[0]), self.__get_tensor(bit_string[1:]), 0).ravel().reshape(2**len(bit_string), -1)

    def __modify_bit_string(self, bit_string):
        res = self.f(bit_string[:-1])
        if res==0:
            return bit_string
        elif res==1:
            if bit_string[-1]=="0":
                return bit_string[:-1]+"1"
            elif bit_string[-1]=="1":
                return bit_string[:-1]+"0"

    def __produce_u_f_gate(self):
        bit_strings = self.__generate_bit_strings(self.n+1)
        xs = list()
        bs = list()
        for bit_string in bit_strings:
            xs.append(self.__get_tensor(bit_string))
            modified_bit_string = self.__modify_bit_string(bit_string)
            bs.append(self.__get_tensor(modified_bit_string))
        
        X = np.hstack(tuple(xs))
        B = np.hstack(tuple(bs))
        
        A = np.linalg.solve(np.linalg.inv(B), np.linalg.inv(X))
        self.__Uf = np.array(A)
        
    def __build_circuit(self):
        self.__produce_u_f_gate()

        circuit = QuantumCircuit(self.n + 1, self.n)

        circuit.x(0)

        for i in range(self.n + 1):
            circuit.h(i)

        circuit.unitary(Operator(self.__Uf), [i for i in range(self.n + 1)], label='Uf')

        for i in range(1, self.n + 1):
            circuit.h(i)

        circuit.measure([i for i in range(self.n, 0, -1)],[i for i in range(self.n)])

        #print(circuit.draw())

        self.__circuit =circuit
    
    def solve(self):
        '''
        Run and measure the quantum circuit
        and return the result.
        '''
        simulator = Aer.get_backend("qasm_simulator")
        job = execute(self.__circuit, simulator, shots=1000)

        # Grab results from the job
        result = job.result()

        # Returns counts
        counts = result.get_counts(self.__circuit)

        print(counts)

        for x, y in counts.items():
            a = x
            break
        
        return a

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

def inner_product_mod_2(x,y):
    '''
    Calculates inner product from two given bit strings
    and then applies mod of 2

    Input: x, y (Bit strings)
    Output: Inner Product mod 2 Ex. x="1011" y="1101" -> (1+0+0+1)%2 = 2%2 = 0.
    '''
    # x, y are bit strings
    sum=0
    for i,j in zip(x,y):
        sum+=int(i)*int(j)
    return sum%2

def addition_mod_2(x,y):
    '''
    Adds two bits and then applied mod 2

    Input: x, y (Bits)
    Output: Sum mod 2 Ex. x=1, y=1 -> (1+1)%2 = 2%2 = 0.
    '''
    #x, y are bits which are integers 0 or 1
    return (x+y)%2

#Test function with a completely randomized value of a and b.
n = 10
#random_bit_string_generator(len(x))
#random.choice([0,1])
f = lambda x: addition_mod_2(inner_product_mod_2("1010111010", x), 1)

solver = Solver(f, n)
a = solver.solve()
print('Value of a: {}'.format(a))
print('Value of b: {}'.format(f("0"*n)))