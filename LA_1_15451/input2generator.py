import sys
import random

output = open('input2.txt', 'w')

n = random.randint(30,30)

print >> output, n

for i in range(n):
	for j in range(n-1):
		print >> output, random.uniform(0,10),
	print >> output, random.uniform(0,10)