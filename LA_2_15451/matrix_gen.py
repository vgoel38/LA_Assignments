import sys
import numpy as np

sys.stdout = open('gram_schimdt_input.txt','w')

A = np.random.randint(100,size=(3, 5))
# A = (A + A.T)/2

for val in A:
	print(*val)