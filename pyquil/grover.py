import numpy as np
import time
import random

from pyquil import get_qc, Program
from pyquil.quil import DefGate
from pyquil.gates import *
from pyquil.api import local_forest_runtime

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
        self.__z_0_definition = DefGate("Z_0", z_0)
        self.__Z_0 = self.__z_0_definition.get_constructor()


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
        self.__z_f_definition = DefGate("Z_f", z_f)
        self.__Z_f = self.__z_f_definition.get_constructor()

    def __produce_negative_gate(self):
        '''
        Produce matrix and gate for changing
        the coefficient of the set of qubits.
        '''
        negative =  -np.identity(2**n)
        self.__negative_definition = DefGate("NEGATIVE", negative)
        self.__NEGATIVE = self.__negative_definition.get_constructor()

    def __build_circuit(self):
        '''
        Build the circuit for Grover's algorithm
        '''
        self.__produce_z_f_gate()
        self.__produce_z_0_gate()
        self.__produce_negative_gate()

        #The part of the Grover's algorithm circuit
        #which might repeated to obtain the correct solution.
        G=Program()
        G += self.__z_f_definition
        G += self.__Z_f(*range(self.n))
        for i in range(self.n):
            G += H(i)
        G+=self.__z_0_definition
        G+=self.__Z_0(*range(self.n))
        for i in range(self.n):
            G += H(i)
        G+=self.__negative_definition
        G+=self.__NEGATIVE(*range(self.n))

        #The main circuit for the algorithm
        self.__p = Program()
        for i in range(self.n):
            self.__p += H(i)
        for i in range(self.k):
            self.__p+=G
    
    @timer
    def solve(self):
        '''
        Run and measure the quantum circuit
        and return the result.
        The circuit is run for n_trials number of trials.
        '''
        with local_forest_runtime():
            qc = get_qc('9q-square-qvm')
            qc.compiler.client.timeout = 10000
            n_trials = 10
            result = qc.run_and_measure(self.__p, trials = self.n_trials)
        values = list()
        for j in range(self.n_trials):
            value = ''
            for i in range(self.n):
                value+=str(result[i][j])
            values.append(value)
        return values

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

n=5
bit_string = random_bit_string_generator(n)

#Test function
f = lambda x: 1 if x==bit_string else 0

solver = Solver(f, n)
xs = solver.solve()

for idx, x in enumerate(xs):
    print("Trial {}, x: {}".format(idx, x))