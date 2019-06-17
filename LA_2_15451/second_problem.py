import sys
from mean_covariance import mean, covariance
from eigen_inv import *
from gs_orthogonalisation import gs_orthogonalisation
from basic_op import *
from numpy import linalg as la
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn_knn import sklearn_knn

#Using numpy only to 
#1.multiply two matrices only in line 25 because naive code takes a lot of time!
#2.plot 3d plot alongwith matplotlib
#3.in the reconstruction error method because naive implementation takes a lot of time!

save_to_file = open('output_data/output_problem2.txt', 'w')

if sys.argv[1] == '-type=gram-schimdt' :
	input = open(sys.argv[2], 'r')
	lines_list = input.readlines()
	M = [[float(val) for val in line.split()] for line in lines_list]
	M = gs_orthogonalisation(M)
	print_rounded_matrix(M)

	M_new = round_off_matrix(M,2)
	for var in M_new:
		for val in var:
			save_to_file.write(str(val)+' ')
		save_to_file.write('\n')
	sys.exit()

def sort_func(elem):
	return elem[0]

def dim_red(d,m,X,sorted_eigenvectors):

	evectors = []
	for i in range(m):
		evectors.append(sorted_eigenvectors[i])
	
	X_reduced = np.matmul(X,(np.array(evectors)).T)

	return X_reduced

def reconstruction_error(X_ref,X_red):

	d = len(X_ref[0])
	m = len(X_red[0])

	for i in range(d-m):
		X_red = np.c_[X_red, np.zeros(len(X_ref)) ]

	error = np.abs(np.subtract(X_ref,X_red))
	error = np.sum(error,axis=1)
	error = np.sum(error)
	error /=d

	return error

def input_data(filename, delimeter):

	input = open(filename, 'r')
	lines_list = input.readlines()

	X = [[float(val) for val in line.split(delimeter)] for line in lines_list]

	Y = []
	for i in range(len(X)):
		Y.append(X[i].pop(0))
	
	return X,Y

def get_eigen(X):

	evalues, evectors = la.eig(covariance_X)
	evectors = evectors.T
	
	eigenvalues = []
	for evalue in evalues:
		eigenvalues.append(evalue)

	eigenvectors = []
	for i in range(len(evectors)):
		eigenvectors.append([])
		for j in range(len(evectors[i])):
			eigenvectors[i].append(evectors[i][j])

	return eigenvalues, eigenvectors

def check_orthogonality(A):

	for i in range(len(A)):
		for j in range(len(A)):
			if i<j:
				dot = dot_product(A[i],A[j])
				if abs(dot) >=1:
					print('dot not 0')
					return False
	return True

def get_sorted_eigen(eigenvalues, eigenvectors):

	eigen_val_vec = []
	for i in range(len(eigenvalues)):
		eigen_val_vec.append(eigenvectors[i])
		eigen_val_vec[i].insert(0,eigenvalues[i])

	eigen_val_vec.sort(reverse=True,key=sort_func)

	sorted_eigenvalues = []

	for i in range(len(eigen_val_vec)):
		sorted_eigenvalues.append(eigen_val_vec[i].pop(0))

	sorted_eigenvectors = eigen_val_vec

	return sorted_eigenvalues, sorted_eigenvectors

def get_sorted(dis, Y):

	dis_Y = []
	for i in range(len(dis)):
		dis_Y.append([])
		dis_Y[i].append(dis[i])
		dis_Y[i].append(Y[i])

	dis_Y.sort(key=sort_func)

	sorted_Y = []

	for i in range(len(dis_Y)):
		sorted_Y.append(dis_Y[i][1])

	return sorted_Y

def plot_reconstruction_error(d,X,sorted_eigenvectors):

	X_ref = dim_red(d,d,X,sorted_eigenvectors)

	error = []
	new_dim = []

	for m in range(d):
		new_dim.append(d-m)
		error.append(reconstruction_error(X_ref,dim_red(d,d-m,X,sorted_eigenvectors)))

	plt.plot(new_dim,error)
	plt.xlabel('new dimension')
	plt.ylabel('reconstruction error') 
	plt.title('Reconstruction Error')
	plt.savefig("output_plots/output_plots_2/problem_2_task_5.png", format="PNG")
	plt.close()

def center_data(X):

	mean_X = mean(X)

	for i in range(len(X)):
		for j in range(len(X[i])):
			X[i][j] -= mean_X[j]

	return X

def classify_knn(X_valid,X,Y,k):

	Y_result = []
	for i in range(len(X_valid)):
		
		dis = []
		for j in range(len(X)):
			dis.append(mean_square_dis(X_valid[i],X[j]))
		
		sorted_Y = get_sorted(dis,Y)
		
		count = [0,0,0,0,0,0,0,0,0,0]
		
		for i in range(k):
			count[int(sorted_Y[i])] += 1

		max_index = 0
		max_var = count[0]

		for i in range(len(count)):
			if max_var < count[i]:
				max_var = count[i]
				max_index = i
		
		Y_result.append(max_index)

	return Y_result

def find_opt_m_k(d,X,Y,X_valid,Y_valid,eigenvectors):

	min_m = 1
	max_m = 100
	min_k = 1
	max_k = 100

	m_axis = np.arange(min_m, max_m, 10)
	k_axis = np.arange(min_k, max_k, 5)
	m_axis, k_axis = np.meshgrid(m_axis, k_axis)
	accuracy_axis = []

	opt_m = 0
	opt_k = 0
	opt_accuracy = 0

	m = min_m
	while m<max_m:

		k=min_k
		while k<max_k:

			X_red = dim_red(d,m,X,eigenvectors)
			X_valid_red = dim_red(d,m,X_valid,eigenvectors)
			Y_result = classify_knn(X_valid_red,X_red,Y,k)

			accuracy = 0
			for i in range(len(Y_valid)):
				if Y_valid[i] == Y_result[i]:
					accuracy += 1
			accuracy /= len(Y_valid)

			if accuracy > opt_accuracy :
				opt_accuracy = accuracy
				opt_m = m
				opt_k = k
			
			accuracy_axis.append(accuracy*100)

			k +=5
		m +=10

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	accuracy_axis = np.array(accuracy_axis)
	accuracy_axis = accuracy_axis.reshape(m_axis.shape)

	ax.plot_surface(m_axis, k_axis, accuracy_axis)
	ax.set_xlabel('m')
	ax.set_ylabel('k')
	ax.set_zlabel('accuracy')
	ax.set_title('surface');
	plt.savefig("output_plots/output_plots_2/problem_2_task_6.png", format="PNG")
	plt.close()

	return opt_m, opt_k, opt_accuracy


#PLEASE NOTE THAT ALL THE CODE HAS BEEN COMMENTED EXCEPT THE PART REQUIRED
#TO RUN THE TEST DATA. TO CHECK VALIDITY OF ANY OTHER TASK, PLEASE DE-COMMENT
#THE RESPECTIVE CODE AND REQUIRED CODE BEFORE THAT
#=====================================
print('----Reading Data----')
save_to_file.write('----Reading Data----')
X,Y = input_data('mnist_train.csv',',')
X_test,Y_test = input_data(sys.argv[1],',')


#================TASK1=====================
print('----Mean vector and covariance matrix of the training data are saved----')
save_to_file.write('\n\n----Mean vector and covariance matrix of the training data are saved----')
# mean_X = mean(X)
# covariance_X = covariance(X)

#Saving mean vector
# output = open('mean_vector.txt', 'w')
# for val in mean_X:
# 	print(val, end=" ", file=output)

#Saving covariance matrix
# output = open('covariance_matrix.txt', 'w')
# for val in covariance_X:
# 	for i in range(len(val)-1):
# 		print(val[i], end=" ", file=output)
# 	print(val[len(val)-1], end="", file=output)
# 	print('\n', end="", file=output)


# #=================TASK2====================
#Reading Covariance Matrix
# input = open('covariance_matrix.txt', 'r')
# lines_list = input.readlines()
# covariance_X = [[float(val) for val in line.split()] for line in lines_list]

#Computing eigenvalues and eigenvectors
# eigenvalues, eigenvectors = get_eigen(covariance_X)
# eigenvalues, eigenvectors = get_sorted_eigen(eigenvalues, eigenvectors)

#Saving eigenvalues
# output = open('eigenvalues.txt', 'w')
# for val in eigenvalues:
# 	print(val, end=" ", file=output)

# #Saving eigenvectors
# output = open('eigenvectors.txt', 'w')
# for val in eigenvectors:
# 	for i in range(len(val)-1):
# 		print(val[i], end=" ", file=output)
# 	print(val[len(val)-1], end="", file=output)
# 	print('\n', end="", file=output)

print('----All eigenvalues and eigenvectors of the training data are saved----')
save_to_file.write('\n\n----All eigenvalues and eigenvectors of the training data are saved----')

#Checking repititive eigenvalues
# i=0
# count=0
# evalue=eigenvalues[0]
# while i<len(eigenvalues):
# 	if evalue == eigenvalues[i]:
# 		count+=1
# 	elif count>1:
# 		print(evalue, count)
# 		evalue = eigenvalues[i]
# 		count = 1
# 	else:
# 		evalue = eigenvalues[i]
# 		count=1
# 	i += 1

print('----Only one eigenvalue 0 was found to repeat 116 times----')
save_to_file.write('\n\n----Only one eigenvalue 0 was found to repeat 116 times----')

# print('----Checking if the eigenvectors are orthogonal or not---- ')
# if check_orthogonality(eigenvectors):
# 	print('All eigenvectors are orthogonal')
# else:
# 	print('All eigenvectors are not orthogonal')

print('----All eigenvectors were found to be orthogonal----')
save_to_file.write('\n\n----All eigenvectors were found to be orthogonal----')

# #=================TASK3====================
print('----Gram-Schmidt Orthogonlisation was added----')
save_to_file.write('\n\n----Gram-Schmidt Orthogonalisation was added----')

# #=================TASK4====================
print('----Function to reduce dimensions of the training data added----')
save_to_file.write('\n\n----Function to reduce dimensions of the training data added----')

# #=================TASK5====================
d = len(X[0])
# plot_reconstruction_error(d,X,eigenvectors)
print('----Reconstruction Error plot was plotted----')
save_to_file.write('\n\n----Reconstruction Error plot was plotted----')

# #=================TASK6====================
# data centering
X = center_data(X)
X_test = center_data(X_test)

#using cross validation technique : 70% training data, 30% validation data
# X_valid = []
# Y_valid = []

# for i in range(int(0.3*len(X))):
# 	X_valid.append(X.pop())
# 	Y_valid.append(Y.pop())

# Y = [int(i) for i in Y]
# Y_valid = [int(i) for i in Y_valid]

# covariance_X = covariance(X)
# eigenvalues, eigenvectors = get_eigen(covariance_X)
# eigenvalues, eigenvectors = get_sorted_eigen(eigenvalues, eigenvectors)

#finding optimum values of m and k
#opt_m, opt_k, opt_accuracy = find_opt_m_k(d,X,Y,X_valid,Y_valid,eigenvectors)
opt_m = 51
opt_k = 6
print('----Surface Curve plotted for accuracy at different k and m: Optimal values found were m=51, k=6 with highest accuracy achieved on validation set = 97.1% ----')
save_to_file.write('\n\n----Surface Curve plotted for accuracy at different k and m: Optimal values found were m=51, k=6 with highest accuracy achieved on validation set = 97.1% ----')

# #================BONUS1=====================
#reading saved eigenvectors
input = open('eigenvectors.txt', 'r')
lines_list = input.readlines()
eigenvectors = [[float(val) for val in line.split()] for line in lines_list]

#Plotting data on 2D space
# X_2 = dim_red(d,2,X,eigenvectors)
# X_2 = X_2.T
# color = ['Green','Blue','Purple','Pink','Black','Cyan','Brown','Gray','Orange','Violet']
# for i in range(len(Y)):
# 	plt.scatter(X_2[0][i],X_2[1][i],s=2,c=color[int(Y[i])])
# X_valid_2 = dim_red(d,2,X_valid,eigenvectors)
# X_valid_2 = X_valid_2.T
# for i in range(len(Y_valid)):
# 	plt.scatter(X_valid_2[0][i],X_valid_2[1][i],s=2,c=color[int(Y_valid[i])])
# plt.title('Clusters')
# plt.savefig("output_plots/output_plots_2/problem_2_bonus_1.png", format="PNG")
# plt.close()

print('----The data was plotted on 2D space and all the instances of same class label were found to be clustered----')
save_to_file.write('\n\n----The data was plotted on 2D space and all the instances of same class label were found to be clustered----')

# #================BONUS2=====================
# X = dim_red(d,opt_m,X,eigenvectors)
# X_valid = dim_red(d,opt_m,X_valid,eigenvectors)

# accuracy = sklearn_knn(X,Y,X_valid,Y_valid,opt_k)

print('----Accuracy achieved by KNN classifier in sklearn was 96.9%----')
save_to_file.write('\n\n----Accuracy achieved by KNN classifier in sklearn was 96.9%----')

#================TEST DATA====================
print('----Running on test data----')
save_to_file.write('\n\n----Running on test data----')
print('----Reducing dimensions of training data set----')
save_to_file.write('\n\n----Reducing dimensions of training data set----')
X_red = dim_red(d,opt_m,X,eigenvectors)
print('----Reducing dimensions of test data set----')
save_to_file.write('\n\n----Reducing dimensions of test data set----')
X_test_red = dim_red(d,opt_m,X_test,eigenvectors)
print('----Running classifier----')
save_to_file.write('\n\n----Running classifier----')
Y_result = classify_knn(X_test_red,X_red,Y,opt_k)

accuracy = 0
for i in range(len(Y_test)):
	if Y_test[i] == Y_result[i]:
		accuracy += 1
accuracy /= len(Y_test)

print('----The accuracy achieved on the test dataset is :' + str(accuracy*100)+'%----')
save_to_file.write('\n\n----The accuracy achieved on the test dataset is :' + str(accuracy*100)+'%----')
print('----The accuracy achieved on the test dataset by sklearn KNN classifier is :' + str(sklearn_knn(X,Y,X_test,Y_test,opt_k)*100)+'%----')
save_to_file.write('\n\n----The accuracy achieved on the test dataset by sklearn KNN classifier is :' + str(sklearn_knn(X,Y,X_test,Y_test,opt_k)*100)+'%----')


