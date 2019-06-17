# -*- coding: utf-8 -*-
import sys
import random

output = open('input1.txt', 'w')

n = random.randint(4,4)
k = random.randint(4,4)

# print >> output, n, k

for i in range(n-1):
	print >> output, random.randint(10,15),
print >> output, random.randint(10,15)

for i in range(n):
	for j in range(k-1):
		print >> output, round(random.uniform(0,0.3),1),
	print >> output, round(random.uniform(0,0.3),1)

for i in range(k):
	print >> output, random.randint(20,30),