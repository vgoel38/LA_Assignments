import sys
import math
from basic_op import *
#numpy used only to compute inverse faster in line 17. Since matrix inverse computation was already a part of assignment1,
#I assume using it here is okay.
import numpy as np

#returns eigenvector corresponding to evalue of A, starting with evector as b
def inverse_power(A, evalue, b):

	iterations = 10
	n = len(A)

	for i in range(iterations):
		# print(i)
		temp = multiply_scalar_matrix(identity_matrix(n),evalue)
		temp = subtract_matrices(A,temp)
		temp = np.linalg.inv(temp)
		temp = multiply_matrices(temp,transpose([b]))
		temp = transpose(temp)
		temp = vectorise(temp)
		b = multiply_scalar(temp,1/norm(temp))

	return b

#returns coefficients of the characteristic polynomial of matrix A
def faddeev_leverrier(A):

	coefficients = []
	B = zero_matrix(len(A))
	coefficients.append(1)
	prod = multiply_matrices(A,B)
	
	for i in range(len(A)):
		B = prod
		for j in range(len(B)):
			B[j][j] += coefficients[i]
		prod = multiply_matrices(A,B)
		coefficients.append(-trace(prod)/(i+1))
	
	return coefficients


#A is a square symmetric matrix
def eigen_inv(A):

	coefficients = faddeev_leverrier(A)
	eigenvalues = np.roots(coefficients)

	eigenvectors = []
	for evalue in eigenvalues:
		eigenvectors.append(inverse_power(A, evalue, random_vector(len(A))))

	return eigenvalues , eigenvectors


if __name__ == "__main__":

	input = open('laplacian_matrix.txt', 'r')
	lines_list = input.readlines()

	A = [[float(val) for val in line.split()] for line in lines_list]

	eigenvalues, eigenvectors = eigen_inv(A)
	print(eigenvalues)
	print_matrix(round_off_matrix(eigenvectors,2))