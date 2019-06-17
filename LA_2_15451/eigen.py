import sys
import math
from gs_orthogonalisation import *
from basic_op import *

def qr_error(A):
	error=0
	for i in range(len(A)):
		for j in range(len(A)):
			if i!=j:
				error = max(abs(A[i][j]),error)
	return error

#A is a square symmetric matrix
def eigen(A):
	#Q and initial A have vectors as rows instead of columns
	#R, A (in the loop) and eigenvectors have vectors as columns instead of rows
	Q = gs_orthogonalisation(A)
	Q = normalise(Q)
	R = multiply_matrices(Q,transpose(A))
	eigenvectors = transpose(Q)

	i = 0

	while 1 :
		A = multiply_matrices(R,transpose(Q))
		Q = gs_orthogonalisation(transpose(A))
		Q = normalise(Q)
		R = multiply_matrices(Q,A)
		eigenvectors = multiply_matrices(eigenvectors, transpose(Q))
		if qr_error(A)<0.0001:
			break
		i+=1

	eigenvalues = []
	i=0
	while i in range(len(A)):
		eigenvalues.append(A[i][i])
		i+=1

	return eigenvalues, transpose(eigenvectors), A


if __name__ == "__main__":

	A = [[2,0],[0,2]]

	# input = open('gram_schimdt_input.txt', 'r')
	# lines_list = input.readlines()

	# A = [[float(val) for val in line.split()] for line in lines_list]

	eigenvalues, eigenvectors, A_morphed = eigen(A)
	# eigenvalues.sort()
	print(eigenvalues)
	print(eigenvectors)