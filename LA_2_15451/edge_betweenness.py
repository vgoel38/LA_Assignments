import sys
from basic_op import *

def bfs(G,s,d):
	
	parent_vertex = []
	
	for i in range(len(G)):
		parent_vertex.append(0)
	parent_vertex[s]=s

	queue = []

	for val in G[s]:
		queue.append(val)
		parent_vertex[val]=s

	while len(queue)>0:
		v = queue.pop(0)
		for val in G[v]:
			if parent_vertex[val] == 0:
				queue.append(val)
				parent_vertex[val]=v

	v=d
	visited_edges=[]
	while parent_vertex[v]!=v:
		visited_edges.append([v,parent_vertex[v]])
		v = parent_vertex[v]

	return visited_edges

def edge_betweenness(G):
	
	centrality_measure = []
	
	for i in range(len(G)):
		centrality_measure.append([])
	for val in centrality_measure:
		for i in range(len(G)):
			val.append(0)

	for i in range(len(G)):
		for j in range(len(G)):
			if i==0 or j==0:
				continue
			if i<j:
				visited_edges = bfs(G,i,j)
				for val in visited_edges:
					centrality_measure[val[0]][val[1]]+=1
					centrality_measure[val[1]][val[0]]+=1

	return centrality_measure