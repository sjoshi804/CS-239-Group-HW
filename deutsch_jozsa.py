import numpy as np
from pyquil import Program, get_qc
from pyquil.gates import *
from pyquil.quil import DefGate
from pyquil.quilatom import unpack_qubit

from typing import Dict

def create_Uf(mappings: Dict[str, str]) -> np.ndarray:
        
        num_qubits = int(np.log2(len(mappings)))
        bitsum = sum([int(bit) for bit in mappings.values()])

        #Checking whether the given mapping is either constant or balanced
        if(not(bitsum == 0 or bitsum == 2 ** (num_qubits - 1) or bitsum == 2 ** num_qubits)):
            raise ValueError("f(x) must be constant or balanced")

        val = 2 ** (num_qubits + 1)

        Uf = np.zeros((val , val)) #Creating a zero matrix of appropriate dimensions initially

        for i in range(val): #Going over all bit strings
            inp = bin(i)[2:].zfill(num_qubits + 1) #Converts integer to bit string of specified length

            x = inp[0:num_qubits]
            fx = mappings[x] #fx is the output of f applied on x

            b = inp[num_qubits] #Helper qubit state initially

            if b == fx:
                bfx = '0'
            else:
                bfx = '1'

            result = x + bfx #This is the resulting qubit states on applying Uf to inp

            row = i
            col = int(result, 2)

            Uf[row][col] = 1

        return Uf


mappings = {}
n = int(input("no. of qubits: "))
print("Enter input, output pairs each in a line:")
for i in range(2 ** n):
    inp, out = input().split()
    mappings[inp] = out

#UfMatrix = [[1, 0, 0, 0, 0, 0, 0, 0],
#            [0, 1, 0, 0, 0, 0, 0, 0],
#            [0, 0, 1, 0, 0, 0, 0, 0],
#            [0, 0, 0, 1, 0, 0, 0, 0],
#            [0, 0, 0, 0, 0, 1, 0, 0],
#            [0, 0, 0, 0, 1, 0, 0, 0],
#            [0, 0, 0, 0, 0, 0, 0, 1],
#            [0, 0, 0, 0, 0, 0, 1, 0]]

UfMatrix = create_Uf(mappings)
print(UfMatrix)

prog = Program()

prog += X(n)

for i in range(n + 1):
    prog += H(i)

u_f_def = DefGate("Uf", UfMatrix)
qubits = [unpack_qubit(i) for i in range(n + 1)]
prog += Program(u_f_def, Gate("Uf", [], qubits))

for i in range(n):
    prog += H(i)

qc_name = "{}q-qvm".format(n + 1)
qc = get_qc(qc_name)
trails = 10
result = qc.run_and_measure(prog, 10)

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
