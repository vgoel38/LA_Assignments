import networkx as nx
import matplotlib.pyplot as plt
from edge_betweenness import edge_betweenness
from basic_op import *
from eigen_inv import eigen_inv
from gs_orthogonalisation import normalise
from normalise_laplacian import normalise_laplacian
#Used only in Bonus1
import numpy as np
from numpy import linalg as la
import warnings

warnings.filterwarnings("ignore")

#NOTE: Assuming nodes start from 1 instead of 0

def sort_func(elem):
	return elem[0]

def variance(A):
	m = sum(A)/len(A)
	var = sum((elem - m) ** 2 for elem in A) / len(A)
	return var

def same_list(l1, l2):
	if len(l1)!=len(l2):
		return False
	l1.sort()
	l2.sort()

	if l1 == l2:
		return True
	else:
	 return False

def get_sorted_eigen(eigenvalues, eigenvectors):

	eigen_val_vec = []
	for i in range(len(eigenvalues)):
		eigen_val_vec.append(eigenvectors[i])
		eigen_val_vec[i].insert(0,eigenvalues[i])

	eigen_val_vec.sort(key=sort_func)

	sorted_eigenvalues = []

	for i in range(len(eigen_val_vec)):
		sorted_eigenvalues.append(eigen_val_vec[i].pop(0))

	sorted_eigenvectors = eigen_val_vec

	return sorted_eigenvalues, sorted_eigenvectors

save_to_file = open('output_data/output_problem1.txt', 'w')

G_nx = nx.read_gml(sys.argv[1])
G_nx = nx.to_undirected(G_nx)

G = [[]]
N = len(G_nx.nodes)

for val in G_nx.nodes:
	G.append([])

for val in G_nx.edges:
		G[int(val[0])].append(int(val[1]))
		G[int(val[1])].append(int(val[0]))

degrees = []

for val in G:
	degrees.append(len(val))

degree_distribution = []

for i in range(len(G)-1):
	degree_distribution.append(0)

for val in degrees[1:]:
	degree_distribution[val] += 1

for i in range(len(degree_distribution)):
	degree_distribution[i] /= N

#==============TASK1================
possible_degrees = [i for i in range(N)]

plt.plot(possible_degrees,degree_distribution)
plt.xlabel('degree')
plt.ylabel('probability') 
plt.title('Degree distribution of graph')
plt.savefig("output_plots/output_plots_1/problem_1_task_1.png", format="PNG")
plt.close()

to_print = '1.The degree distribution of the nodes has been plotted. It resembles Binomial distribution, which can turn into Poisson for more number of nodes.\n'
print(to_print)
save_to_file.write('\n\n'+to_print)


#==============TASK2================
nodes=[]
for i in range(N+1):
	nodes.append(i)

plt.bar(nodes[1::], degrees[1::], tick_label = nodes[1::], width = 0.8, color = ['red']) 
plt.xlabel('nodes') 
plt.ylabel('centrality') 
plt.title('Degree centrality of each node in the graph')
plt.savefig("output_plots/output_plots_1/problem_1_task_2.png", format="PNG")
plt.close()

top_degree = -1
top_node = -1
second_top_degree = -1
second_top_node = -1
i=1
while i<len(nodes):
	if degrees[i]>top_degree:
		second_top_degree = top_degree
		second_top_node = top_node
		top_degree = degrees[i]
		top_node = i
	i+=1

to_print = '2.The histogram has been plotted. The top two central nodes are '+str(top_node)+' and '+str(second_top_node)+'\n'
print(to_print)
save_to_file.write('\n\n'+to_print)


#==============TASK3================
edge_centrality_matrix = edge_betweenness(G)

top_edge_val = -1
top_v1 = -1
top_v2 = -1

i=1
while i < len(edge_centrality_matrix):
	j=1
	while j < len(edge_centrality_matrix):
		if top_edge_val < edge_centrality_matrix[i][j]:
			top_edge_val = edge_centrality_matrix[i][j]
			top_v1 = i
			top_v2 = j
		j+=1
	i+=1

to_print = '3.The most central edge is the edge between nodes '+str(top_v1)+' and '+str(top_v2)+'\n'
print(to_print)
save_to_file.write('\n\n'+to_print)

#==============TASK4================
adj_matrix = []

for i in range(len(G)):
	adj_matrix.append([])
	for j in range(len(G)):
		adj_matrix[i].append(0)

for i in range(len(G)):
	if i==0:
		continue
	for val in G[i]:
		adj_matrix[i][val]=1

deg_matrix = []

for i in range(len(G)):
	deg_matrix.append([])
	for j in range(len(G)):
		deg_matrix[i].append(0)

for i in range(len(degrees)):
	deg_matrix[i][i]=degrees[i]

L = subtract_matrices(deg_matrix, adj_matrix)
L.pop(0)
for val in L:
	val.pop(0)

L_normalised = normalise_laplacian(L)

output = open('laplacian_matrix.txt', 'w')
for val in L_normalised:
	for i in range(len(val)-1):
		print(val[i], end=" ", file=output)
	print(val[len(val)-1], end="", file=output)
	print('\n', end="", file=output)

output.close()

eigenvalues, eigenvectors = eigen_inv(L_normalised)

to_print = '4.All eigenvalues, as follows, are real.'
print(to_print)
print(eigenvalues)
print('\n')
save_to_file.write('\n\n'+to_print)
save_to_file.write('\n'+str(eigenvalues))
save_to_file.write('\n')


#=================TASK5==================
eigenvalues, eigenvectors = get_sorted_eigen(eigenvalues, eigenvectors)
print('5.The two smallest eigenvalues are '+str(eigenvalues[0])+' and '+str(eigenvalues[1]))
print('The corresponding eigenvectors are as follows')
print(round_off_vector(eigenvectors[0]))
print(round_off_vector(eigenvectors[1]))
print('Error in smallest eigenvalue: '+str(abs(eigenvalues[0])))

save_to_file.write('\n\n5.The two smallest eigenvalues are '+str(eigenvalues[0])+' and '+str(eigenvalues[1]))
save_to_file.write('\n\nThe corresponding eigenvectors are as follows\n')
save_to_file.write('\n'+str(round_off_vector(eigenvectors[0]))+'\n')
save_to_file.write('\n'+str(round_off_vector(eigenvectors[1])))
save_to_file.write('\nError in smallest eigenvalue: '+str(abs(eigenvalues[0])))

mean_evector = 0
for elem in eigenvectors[0]:
	mean_evector += abs(elem)
mean_evector = mean_evector/len(eigenvectors[0])
mean_error = 0
for elem in eigenvectors[0]:
	mean_error += abs(abs(elem) - abs(mean_evector))
mean_error = mean_error/len(eigenvectors[0])

to_print = '\nSince all values in eigenvector corresponding to smallest eigenvalue should be equal, error: '+str(round(mean_error*100))+'%\n'
print(to_print)
save_to_file.write(to_print)

#===================TASK6================
node_color = []

for i in range(N):
	node_color.append('r')

sorted_eigenvalues = []
for elem in eigenvalues:
	sorted_eigenvalues.append(elem)
sorted_eigenvalues.sort()

cluster1 = []
cluster2 = []
var1 = []
var2 = []

for i in range(len(eigenvalues)):
	if eigenvalues[i] == sorted_eigenvalues[1]:
		for j in range(len(eigenvectors[i])):
			if eigenvectors[i][j]<0:
				cluster2.append(j)
				node_color[j]='b'
				var2.append(eigenvectors[i][j])
			else:
				cluster1.append(j)
				var1.append(eigenvectors[i][j])
		break

to_print = '6.The two clusters got are (say House A and House B respectively):'
print(to_print)
print(cluster1)
print(cluster2)
save_to_file.write('\n\n'+to_print)
save_to_file.write('\n'+str(cluster1))
save_to_file.write('\n'+str(cluster2))


nx.draw(G_nx,with_labels=True,font_size=10,node_color=node_color)
plt.savefig("output_plots/output_plots_1/problem_1_task_6.png", format="PNG")
plt.close()
to_print = 'The nodes were coloured using eigenvector wrt the second smallest eigenvalue and the plot was saved. The graph got divided into two clusters that were actually houses.\n'
print(to_print)
save_to_file.write('\n\n'+to_print)


#===================TASK7================
if (variance(var1)>variance(var2)):
	to_print = '7. The variability of House A is more than that of House B. Thus the House of Algebrician should befriend House A\n'
else:
	to_print = '7. The variability of House B is more than that of House A. Thus the House of Algebrician should befriend House B\n'

print(to_print)
save_to_file.write('\n\n'+to_print)


#===================BONUS1================
A = np.loadtxt('laplacian_matrix.txt', delimiter=' ')
evalues, evectors = la.eig(A)
evectors = evectors.T

sorted_evalues = []
for elem in evalues:
	sorted_evalues.append(elem)
sorted_evalues.sort()

np_cluster1 = []
np_cluster2 = []

for i in range(len(evalues)):
	if evalues[i] == sorted_evalues[1]:
		for j in range(len(evectors[i])):
			if evectors[i][j]<0:
				node_color[j]='b'
				np_cluster1.append(j)
			else:
				np_cluster2.append(j)
		break

to_print = 'B1. The clusters formed by numpy are'
print(to_print)
print(np_cluster1)
print(np_cluster2)
save_to_file.write('\n\n'+to_print)
save_to_file.write('\n'+str(np_cluster1))
save_to_file.write('\n'+str(np_cluster1))

if (same_list(cluster1, np_cluster1) and same_list(cluster2, np_cluster2)) or (same_list(cluster1, np_cluster2) and same_list(cluster2, np_cluster1)):
	to_print = 'The clusters are exactly same as we got!!\n'
else:
	to_print = 'The clusters are not exactly same!\n'

print(to_print)
save_to_file.write('\n\n'+to_print)

#===================BONUS2================
to_print = 'B2. To find more than two clusters, we can follow this approach:'
print(to_print)
save_to_file.write('\n\n'+to_print)
to_print = 'a. After getting the two clusters, we could apply k-means clustering to make more clusters.'
print(to_print)
save_to_file.write('\n\n'+to_print)
to_print = 'b. After getting the two clusters, we could pick the cluster with more variance and then limiting our graph to only this cluster, we could run our current algorithm to divide this into two clusters. We can repeat the algorithm on the three clusters and find more clusters similarly. The correctness of the idea is obviously questionable but it could be tried out.'
print(to_print)
save_to_file.write('\n\n'+to_print)
to_print = 'c. We could also pick “central” edges in the graph to look for clusters. We can delete the edges with the highest edge centrality and the connected components obtained can be treated as clusters. But this approach is limited to getting random number of clusters instead of a fixed number of clusters.\n'
print(to_print)
save_to_file.write('\n\n'+to_print)

#===================BONUS3================
to_print = 'B3. The clusters correspnoding to the eigenvector wrt 2nd largest eigenvalue has been drawn. As observed, the nodes of two different colours are scattered among each other and thus, do not really make separate clusters.'
print(to_print)
save_to_file.write('\n\n'+to_print)

node_color = []
for i in range(N):
	node_color.append('r')

sorted_eigenvalues = []
for elem in eigenvalues:
	sorted_eigenvalues.append(elem)
sorted_eigenvalues.sort()

for i in range(len(eigenvalues)):
	if eigenvalues[i] == sorted_eigenvalues[len(sorted_eigenvalues)-2]:
		for j in range(len(eigenvectors[i])):
			if eigenvectors[i][j]<0:
				node_color[j]='b'
		break

nx.draw(G_nx,with_labels=True,font_size=10,node_color=node_color)
plt.savefig("output_plots/output_plots_1/problem_1_bonus_3.png", format="PNG")
plt.close()
print('\n')

#===================BONUS4================
to_print = 'B4.The central edge seems to be the one that connects the two clusters.\n'
print(to_print)
save_to_file.write('\n\n'+to_print)


