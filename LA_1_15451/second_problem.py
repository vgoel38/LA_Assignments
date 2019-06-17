# -*- coding: utf-8 -*-
import sys

#input stored in lines_list
input = open(sys.argv[1], 'r')
lines_list = input.readlines()

#output file
output = open('output_problem2.txt', 'w')

#matrix size
n = (int(lines_list[0]))

#matrix
M = [[float(val) for val in line.split()] for line in lines_list[1:]]

#list of operations to convert M to I and then to Minverse
#[En]....[E2][E1][M] = I
#[En]....[E2][E1][I] = Minverse
Operations = []

str_success = 'YAAY! FOUND ONE!'
str_failure = 'ALAS! DIDN’T FIND ONE!'

#used in Switch to check whether pivot exists in a column
allZeroesInCol = True
inverseExists = True


#converting M to Identity Matrix
for i in range(n):

	#Switch
	if M[i][i] == 0:
		for j in range(i+1,n):
			if M[j][i] != 0:
				for k in range(n):
					#switch row i and j
					temp = M[i][k] ; M[i][k]=M[j][k] ; M[j][k]=temp
				allZeroesInCol = False
				Operations.append(['S',i,j,-1])
				break
		if allZeroesInCol :
			inverseExists = False
			break
	allZeroesInCol = True


	#Reduce pivot to 1
	if M[i][i] != 1 :
		Divisor = M[i][i]
		for j in range(i,n):
			M[i][j] /= Divisor
		Operations.append(['M',Divisor,i,-1])

	#Reduce entries of pivot column to 0
	for j in range(n):
		if j!=i and M[j][i]!=0:
			Subtractor = M[j][i]
			for k in range(i,n):
				M[j][k] -= Subtractor*M[i][k]
			Operations.append(['MA',Subtractor,i,j])


if inverseExists:
	#print output to output file
	print >> output, str_success
	#print output to terminal
	print(str_success)

else :
	print >> output, str_failure
	print(str_failure)


if inverseExists:

	#Creating identity matrix of size n 
	I = []
	for i in range(n):
		J = [0]*n
		J[i]=1
		I.append(J)

	#converting identity matrix to Minverse
	for i in range(len(Operations)):

		if Operations[i][0] == 'S':
			for k in range(n):
					temp = M[Operations[i][1]][k]
					M[Operations[i][1]][k] = M[Operations[i][2]][k]
					M[Operations[i][2]][k] = temp

		elif Operations[i][0] == 'M':
			for j in range(n):
				M[Operations[i][2]][j] /= Operations[i][1]

		elif Operations[i][0] == 'MA':
			for k in range(n):
				M[Operations[i][3]][k] -= Operations[i][1]*M[Operations[i][2]][k]

	#rounding off entries of M
	for i in range(n):
	 	for j in range(n):
	 		M[i][j] = round(M[i][j],3)

	#printing Minverse
	if n == 1:
		print >> output, M[0][0]
		print(M[0][0])
	else:
		for i in range(n):
			for j in range(n-1):
				print >> output, M[i][j],
				print(M[i][j]),
			print >> output, M[i][j+1]
			print(M[i][j+1])



#printing operations for M to I to Minverse
for p in range(int(inverseExists)+1):
	for i in range(len(Operations)):
		if Operations[i][0] == 'S':
			print >> output, 'SWITCH %s %s' %(Operations[i][1]+1, Operations[i][2]+1)
			print('SWITCH %s %s' %(Operations[i][1]+1, Operations[i][2]+1))
		elif Operations[i][0] == 'M':
			print >> output, 'MULTIPLY %s %s' %(round(1/Operations[i][1],3), Operations[i][2]+1)
			print('MULTIPLY %s %s' %(round(1/Operations[i][1],3), Operations[i][2]+1))
		elif Operations[i][0] == 'MA':
			print >> output, 'MULTIPLY&ADD %s %s %s' %(round(-Operations[i][1],3), Operations[i][2]+1, Operations[i][3]+1)
			print('MULTIPLY&ADD %s %s %s' %(round(-Operations[i][1],3), Operations[i][2]+1, Operations[i][3]+1))

