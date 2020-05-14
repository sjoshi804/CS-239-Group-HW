import numpy as np

from pyquil import get_qc, Program
from pyquil.quil import DefGate
from pyquil.gates import *
from pyquil.api import local_forest_runtime


class Solver(object):

    def __init__(self, f, n, n_trials=10):   
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
        z_0 = np.identity(2**n)
        z_0[0][0] = -z_0[0][0]
        self.__z_0_definition = DefGate("Z_0", z_0)
        self.__Z_0 = self.__z_0_definition.get_constructor()


    def __produce_z_f_gate(self):
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
        negative =  -np.identity(2**n)
        self.__negative_definition = DefGate("NEGATIVE", negative)
        self.__NEGATIVE = self.__negative_definition.get_constructor()

    def __build_circuit(self):
        self.__produce_z_f_gate()
        self.__produce_z_0_gate()
        self.__produce_negative_gate()

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

        self.__p = Program()
        for i in range(self.n):
            self.__p += H(i)
        for i in range(self.k):
            self.__p+=G
    
    def solve(self):
        with local_forest_runtime():
            qc = get_qc('9q-square-qvm')
            n_trials = 10
            result = qc.run_and_measure(self.__p, trials = self.n_trials)
        values = list()
        for j in range(self.n_trials):
            value = ''
            for i in range(self.n):
                value+=str(result[i][j])
            values.append(value)
        return values
n=3
f = lambda x: 1 if x=="101" else 0

solver = Solver(f, n)
xs = solver.solve()
print(xs)
for idx, x in enumerate(xs):
    print("Trial {}, x: {}".format(idx, x))