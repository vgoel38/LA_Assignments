import sys
import math
from basic_op import * 

#NOTE : Each vector is represented as a list and each matrix is represented as a list of lists (each row is a vector)

#projection of vector B on vector A
def projection(B, A):
	dot = dot_product(A,A)
	if dot == 0:
		return [0 for val in A]
	return multiply_scalar(A, dot_product(B,A)/dot_product(A,A))

def check_threshold(v):
	threshhold = 1e-10
	if norm(v)<threshhold:
		for i in range(len(v)):
				if abs(v[i])<threshhold:
					v[i]=0
	return v

#returns M after orthogonalising vectors (note: vectors returned are not orthonormal)
def gs_orthogonalisation(M):

	M_orthogonal = []
	i=0

	while i < len(M):
		v = M[i]
		j=0
		while j<i:
			v = subtract_vectors(v, projection(M[i],M_orthogonal[j]))
			j+=1
		v = check_threshold(v)
		M_orthogonal.append(v)
		i+=1

	return M_orthogonal


if __name__ == "__main__":

	#input stored in lines_list
	input = open('gram_schimdt_input.txt', 'r')
	lines_list = input.readlines()

	M = [[float(val) for val in line.split()] for line in lines_list]

	print_rounded_matrix(gs_orthogonalisation(M))
