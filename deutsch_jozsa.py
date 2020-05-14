import numpy as np
import time
import random

from pyquil import Program, get_qc
from pyquil.gates import *
from pyquil.quil import DefGate
from pyquil.quilatom import unpack_qubit
from pyquil.api import local_forest_runtime

from typing import Dict

#Converts integer to bit string of specified length
def get_bit_string(num, lngt):
	return bin(num)[2:].zfill(lngt)

class DeutschJozsa:

	def __init__(self, mapping):
		self.__mapping = mapping
		self.__num_qubits = int(np.log2(len(mappings)))
		self.__u_f_dim = 2 ** (self.__num_qubits + 1)
		self.__Uf = self.__create_Uf()

	def get_Uf(self):
		return self.__Uf

	def __create_Uf(self) -> np.ndarray:
	        
	        
	        bitsum = sum([int(bit) for bit in self.__mapping.values()])

	        #Checking whether the given mapping is constant or balanced
	        if(not(bitsum == 0 or bitsum == 2 ** (self.__num_qubits - 1) or bitsum == 2 ** self.__num_qubits)):
	            raise ValueError("The function must be constant or balanced")

	        Uf = np.zeros((self.__u_f_dim , self.__u_f_dim)) #Creating a zero matrix of appropriate dimensions initially

	        for i in range(2 ** (self.__num_qubits + 1)): #Going over all bit strings of length num_qubits + 1
	            inp = get_bit_string(i, self.__num_qubits + 1)

	            x = inp[0:self.__num_qubits]
	            fx = mappings[x] #fx is the output of f applied on x

	            b = inp[self.__num_qubits] #Helper qubit state initially

	            if b == fx:
	                bfx = '0' #b^f(x)
	            else:
	                bfx = '1'

	            result = x + bfx #This is the resulting qubit states on applying Uf to inp

	            row = int(result, 2) #Converting the bitstring to int
	            col = i

	            Uf[row][col] = 1 #Generating Uf based on the mapping Uf|x>|b> = |x>|b^f(x)>

	        return Uf


#Testing
mappings = {}
n = int(input("no. of qubits: "))
print("Type 'y' if you want to give input-output pairs as input or type 'n' : ", end='')
ir = input()
if ir == 'y':
	print("Enter space seperated input, output pairs each in a line:")
	for i in range(2 ** n):
	    inp, out = input().split()
	    mappings[inp] = out
else:
	print("Type 'c' if you want a constant function as input or type 'b' for a random balanced function : ", end='')
	cb = input()
	if cb == 'c':
		const_val = str(random.randint(0,1)) #Setting the output to either bit 0 always or 1 always
		for i in range(2 ** n):
			mappings[get_bit_string(i, n)] = const_val
	else:
		#Choosing a random position in the bit string and setting its value as output which will be balanced
		rand_val = random.randint(0,n - 1)
		for i in range(2 ** n):
			bit_str = get_bit_string(i, n)
			mappings[bit_str] = bit_str[rand_val]


UfMatrix = DeutschJozsa(mappings).get_Uf()

#Creating the program
prog = Program()

prog += X(n) #Setting helper qubit state to 1

for i in range(n + 1):
    prog += H(i) #Applying Hadamard to all qubits

u_f_def = DefGate("Uf", UfMatrix)
qubits = [unpack_qubit(i) for i in range(n + 1)]
prog += Program(u_f_def, Gate("Uf", [], qubits)) #Applying Uf

for i in range(n):
    prog += H(i) #Applying Hadamard to computational qubits i.e without helper bit

qc_name = "{}q-qvm".format(n + 1)
trails = 10

with local_forest_runtime():
	time.sleep(1)
	qc = get_qc(qc_name)
	qc.compiler.client.timeout = 20000
	result = qc.run_and_measure(prog, 1) #Trails is currently set to 1

isConstant = True

print("State of qubits without helper qubit in each trail")
for j in range(trails):
    print(j, ": ", end='')
    for i in range(n):
        if(result[i][j] != 0):
            isConstant = False
        print(result[i][j],end='')
    print()

if(isConstant):
    print("Constant")
else :
    print("Balanced")
