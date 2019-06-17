# -*- coding: utf-8 -*-
import sys
import random

#input stored in lines_list
input = open(sys.argv[2], 'r')
lines_list = input.readlines()

#n = number of students, k = number of ingredients, p = potion amounts, M = matrix of percentages, m = max potion amount
if sys.argv[1] == '-part=one':
	n = 4
	k = 4
	p = [float(val) for val in lines_list[0].split()]
	M = [[float(val) for val in line.split()] for line in lines_list[1:]]
	m = M.pop()
	#output file
	output = open('output_problem1_part1.txt', 'w')
else:
	n = int((lines_list[0].split())[0])
	k = int((lines_list[0].split())[1])
	p = [float(val) for val in lines_list[1].split()]
	M = [[float(val) for val in line.split()] for line in lines_list[2:]]
	m = M.pop()
	#output file
	output = open('output_problem1_part2.txt', 'w')

str_no_solution = 'NOT POSSIBLE, SNAPE IS WICKED!'
str_one_solution = 'EXACTLY ONE!'
str_many_solutions = 'MORE THAN ONE!'

#checking whether sum of each column is <= 1
solutionExists = True
for i in range(k):
	sum = 0
	for j in range(n):
		sum += M[j][i]
	if sum > 1:
		solutionExists = False
		break


if solutionExists == False:
	print >> output, str_no_solution
	print(str_no_solution)

else:
	#augmenting M
	for i in range(n):
		M[i].append(p[i])


	#converting M to reduced row echelon form
	c = 0
	r = 0
	while r < n and c < k+1:

		#Switch
		if M[r][c] == 0:
			for i in range(r+1,n):
				if M[i][c] != 0:
					for j in range(k+1):
						temp = M[r][j] ; M[r][j]=M[i][j] ; M[i][j]=temp
					break


		#Reduce pivot to 1
		if M[r][c] != 0:
			if M[r][c] != 1 :
				Divisor = M[r][c]
				for i in range(c,k+1):
					M[r][i] /= Divisor
		#find pivot in next column but same row
		else:
			c +=1
			continue


		#Reduce entries of pivot column to 0
		for i in range(n):
			if i!=r and M[i][c]!=0:
				Subtractor = M[i][c]
				for j in range(c,k+1):
					M[i][j] -= Subtractor*M[r][j]


		#find pivot in next col and next row
		r +=1
		c +=1


	#checking whether any rows have all column values equal to 0 except the last column i.e. no solution condition
	allZeroes = True
	for i in range(n):
		for j in range(k):
			if M[i][j]!=0:
				allZeroes = False
				break
		if allZeroes and M[i][k]!=0:
			solutionExists = False
			break
		allZeroes = True

	if solutionExists == False:
		print >> output, str_no_solution
		print(str_no_solution)
	else:

		#remove all those rows from M that have all column values equal to 0
		allZeroes = True
		i=0
		while i<n:
			for j in range(k):
				if M[i][j] != 0:
					allZeroes = False
					break
			if allZeroes and n > 1:
				M.remove(M[i])
				n -= 1
			else:
				i+=1
			allZeroes = True

		#checking whether all columns are pivot columns
		allONes = True
		for i in range(n):
			if M[i][i]!=1:
				allONes = False
				break

		#looking for a unique solution
		if allONes and n==k:
			#checking whether the unique solution is valid
			solution = []
			for i in range(n):
				solution.append(round(M[i][k],3))
				if M[i][k] > m[i] or M[i][k] < 0:
					solutionExists = False

			if(solutionExists):
				print >> output, str_one_solution
				print(str_one_solution)
				for val in solution:
					print >> output, val,
					print(val),
			else:
				print >> output, str_no_solution
				print(str_no_solution)


		#looking for infinite solutions
		else:

			print >> output, str_many_solutions
			print(str_many_solutions)

			#tells whether i is a pivot or not
			is_pivot = []
			i=0
			for j in range(k):
				if i<n and M[i][j] == 1:
					is_pivot.append(True)
					i+=1
				else:
					is_pivot.append(False)


			particular_sol_found = True
			trials = 0

			#randomly looking for a valid solution
			while trials < 10000 :

				trials +=1

				particular_sol = []
				for i in range(k):
					if is_pivot[i] == False:
						particular_sol.append(random.randint(0,m[i]))
					else:
						particular_sol.append(0)

				i=0
				for j in range(k):
					if i<n and M[i][j] == 1:
						particular_sol[i] = round(M[i][k],3)
						l=j+1
						while l<k:
							if M[i][l]!=0:
								particular_sol[i] -= round(M[i][l],3)*particular_sol[l]
							l+=1
						i+=1

				particular_sol_found = True

				for i in range(k):
					if particular_sol[i]<0 or particular_sol[i]>m[i]:
						particular_sol_found = False
						break

				if particular_sol_found == True:
					break

			#printing one particular solution
			if particular_sol_found == True:
				for i in range(k-1):
					print >> output, round(particular_sol[i],3),
					print round(particular_sol[i],3),
				print >> output, round(particular_sol[k-1],3)
				print round(particular_sol[k-1],3)
			else:
				print >> output, 'Unable to find a valid solution!'
				print 'Unable to find a valid solution!'



			#printing general solution
			i=0
			for j in range(k):
				if i<n and M[i][j] == 1:
					sol = 'I'+str(j+1)+'='+str(round(M[i][k],3))
					particular_sol[i] = round(M[i][k],3)
					l=j+1
					while l<k:
						if M[i][l]!=0:
							sol += '-('+str(round(M[i][l],3))+')*I' + str(l+1)
							particular_sol[i] -= round(M[i][l],3)*particular_sol[l]
						l+=1
					print >> output, sol
					print(sol)
					i+=1


