import sys
from basic_op import *

def mean(M):
	
	mean = []
	for i in range(len(M[0])):
		summation = 0
		for j in range(len(M)):
			summation += M[j][i]
		mean.append(summation/len(M))
	return mean

def covariance(M):
	
	mean_vector = mean(M)

	for i in range(len(M)):
		for j in range(len(M[i])):
			M[i][j] -= mean_vector[j]

	covariance_matrix = multiply_matrices(transpose(M),M)
	covariance_matrix = multiply_scalar_matrix(covariance_matrix,1/(len(M)-1))

	return covariance_matrix



if __name__ == "__main__":

	input = open('gram_schimdt_input.txt', 'r')
	lines_list = input.readlines()

	A = [[float(val) for val in line.split()] for line in lines_list]

	print(mean(A))
	print(covariance(A))