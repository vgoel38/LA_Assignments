import sys
import random
import math

#NOTE : Each vector is represented as a list and each matrix is represented as a list of lists (each row is a vector)

def vectorise(A):
	v = []
	for i in range(len(A[0])):
		v.append(A[0][i])
	return v

def identity_matrix(n):
	I = [[0 for i in range(n)] for j in range(n)]
	for i in range(n):
		I[i][i]=1
	return I

def normalise(M):
	i=0
	while i<len(M):
		dot = dot_product(M[i],M[i])
		if dot != 0:
			M[i] = multiply_scalar(M[i],1/math.sqrt(dot))
		i+=1

	return M

def norm(v):
	result = 0
	for val in v:
		result += val*val

	return math.sqrt(result)

def zero_matrix(n):
	return [[0 for i in range(n)] for j in range(n)]

def zero_vector(n):
	return [0 for i in range(n)]

#returns a random vector of size n
def random_vector(n):
	return [random.uniform(0,1) for i in range(n)]

#A and B are vectors
def subtract_vectors(A,B):
	result = []
	for i in range(len(A)):
		result.append(A[i]-B[i])
	return result

def trace(A):
	result = 0
	for i in range(len(A)):
		result += A[i][i]
	return result

#A is a vector
def multiply_scalar(A, multiplier):
	return [val*multiplier for val in A]

#A and B are vectors
def dot_product(A,B):
	result = 0
	for val1,val2 in zip(A,B):
		result+=val1*val2
	return result

#A is a matrix
def round_off_matrix(A, precision):
	return [[round(elem,2) for elem in vector] for vector in A]

#A is a matrix
def transpose(A):
	AT = []
	for i in range(len(A[0])):
		AT.append([])
	for val in A:
		i=0
		while i<len(val):
			AT[i].append(val[i])
			i+=1
	return AT

#A and B are matrices
def multiply_matrices(A,B):
	B = transpose(B)
	return [[ dot_product(A[i],B[j]) for j in range(len(B))] for i in range(len(A))]

#A and B are matrices
def add_matrices(A,B):
	return [[ A[i][j]+B[i][j] for j in range(len(A))] for i in range(len(A))]

#A and B are matrices
def subtract_matrices(A,B):
	return [[ A[i][j]-B[i][j] for j in range(len(A[0]))] for i in range(len(A))]

#A is a matrix
def multiply_scalar_matrix(A,multiplier):
	return [[ A[i][j]*multiplier for j in range(len(A[i]))] for i in range(len(A))]

#M is a matrix
def print_matrix(M):
	for val in M:
		print(*val)

def print_rounded_matrix(M):
	M_new = round_off_matrix(M,2)
	for val in M_new:
		print(*val)

def round_off_vector(M):
	return [round(elem,2) for elem in M]

def mean_square_dis(A,B):
	dis = 0
	for i in range(len(A)):
		dis += (A[i]-B[i])*(A[i]-B[i])
	return math.sqrt(dis)
	

if __name__ == "__main__":
	A = [1,2,3,4,5]
	B = [10,9,8,7,6]
	M = [[1,2],[10,9]]
	N = [[2,3],[4,5]]

	print(subtract_vectors(A,B))
	print(multiply_scalar(A,2))
	print(dot_product(A,B))
	print(transpose(M))
	print(multiply_matrices(transpose([A]),[B]))
	print(add_matrices(M,N))
	print(multiply_scalar_matrix(M,2))
	print_matrix(M)
	print(identity_matrix(3))
	# print(vectorise([[1,2,3]]))