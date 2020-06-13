from qiskit import QuantumCircuit, execute, Aer
from qiskit.quantum_info.operators import Operator
from qiskit import IBMQ

import numpy as np
import time
import random

#Converts integer to bit string of specified length
def get_bit_string(num, lngt):
	return bin(num)[2:].zfill(lngt)

class DeutschJozsa:

	def __init__(self, mapping):
		self.__mapping = mapping
		self.__num_qubits = int(np.log2(len(mappings)))
		self.__u_f_dim = 2 ** (self.__num_qubits + 1)
		self.__Uf = self.__create_Uf()
		self.__circuit = self.__build_circuit()

	def __create_Uf(self) -> np.ndarray:

		bitsum = sum([int(bit) for bit in self.__mapping.values()])

		#Checking whether the given mapping is constant or balanced
		if(not(bitsum == 0 or bitsum == 2 ** (self.__num_qubits - 1) or bitsum == 2 ** self.__num_qubits)):
			raise ValueError("The function must be constant or balanced")

		Uf = np.zeros((self.__u_f_dim , self.__u_f_dim)) #Creating a zero matrix of appropriate dimensions initially

		for i in range(self.__u_f_dim): #Going over all bit strings of length num_qubits + 1
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

	def __check_identity(self):
		for row in range(self.__u_f_dim) :
			for col in range(self.__u_f_dim) :
				if row == col and self.__Uf[row][col] == 0:
					return False
				if row != col and self.__Uf[row][col] == 1:
					return False
		return True

	def __build_circuit(self):

		circuit = QuantumCircuit(self.__num_qubits + 1, self.__num_qubits)

		circuit.x(0)

		for i in range(self.__num_qubits + 1):
			circuit.h(i)

		if self.__check_identity() == False:
			circuit.unitary(Operator(self.__Uf), [i for i in range(self.__num_qubits + 1)], label='Uf')

		for i in range(1, self.__num_qubits + 1):
			circuit.h(i)

		circuit.measure([i for i in range(self.__num_qubits, 0, -1)],[i for i in range(self.__num_qubits)])
		#print(circuit.draw())

		return circuit

	def solve(self):
		'''
		Run and measure the quantum circuit
		and return the result.
		'''
		# Using an actual quantum computer
		API_CREDENTIAL = "10809dd9b130e194f8d81473a9633f4d563b84f74fb9afe02b03e7703a14591fd8ce43c73f9f4c03dc9f5829ad8cf78a19794084199a1a70a0d61708d0bd272f"
		IBMQ.save_account(API_CREDENTIAL, overwrite=True)
		IBMQ.load_account()
		provider = IBMQ.get_provider(hub='ibm-q')
		qc = provider.get_backend("ibmq_london")
		job = execute(self.__circuit, qc, shots=1000)

		# Grab results from the job
		result = job.result()

		# Returns counts
		counts = result.get_counts(self.__circuit)
		print("No. of times each state appears:",counts)
		time_taken = result.time_taken
		print("Time Taken: ",time_taken)

		return counts

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
		'''Setting 0's and 1's randomly for each bit string and when 
		the count of 0 or 1 reaches half, remaining values are set such that it is balanced.'''
		zero_ct = 0
		one_ct = 0
		half = 2 ** (n - 1)
		for i in range(2 ** n):
			bit_str = get_bit_string(i, n)
			if(zero_ct == half):
				mappings[bit_str] = "1"
			elif(one_ct == half):
				mappings[bit_str] = "0"
			else:
				rand_val = str(random.randint(0,1))
				mappings[bit_str] = rand_val
				if(rand_val == "0"):
					zero_ct += 1
				else:
					one_ct += 1


counts = DeutschJozsa(mappings).solve()

max_freq = 0
for x, y in counts.items():
    if y > max_freq:
        a = x
        y = max_freq

if a == "0"*n:
	print("Constant")
else:
	print("Balanced")

