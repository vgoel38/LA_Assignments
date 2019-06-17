import sys
import math
from basic_op import *

def normalise_laplacian(L):
	
	L_normalised=[]
	
	for i in range(len(L)):
		L_normalised.append([])
	for val in L_normalised:
		for i in range(len(L)):
			val.append(0)

	for i in range(len(L)):
		for j in range(len(L)):
			if i==j and L[i][j]!=0:
				L_normalised[i][j]=1
			elif L[i][j]!=0 and L[j][j]!=0:
				L_normalised[i][j] = -1/(math.sqrt(L[i][i]*L[j][j]))

	return L_normalised

if __name__ == "__main__":

	input = open('laplacian_matrix.txt', 'r')
	lines_list = input.readlines()

	L = [[float(val) for val in line.split()] for line in lines_list]

	print_matrix(normalise_laplacian(L))
