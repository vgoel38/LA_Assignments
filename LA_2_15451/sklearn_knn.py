from sklearn.neighbors import KNeighborsClassifier

def sklearn_knn(X,Y,X_valid,Y_valid,k):

	knn = KNeighborsClassifier(k)
	knn.fit(X,Y)
	accuracy = knn.score(X_valid,Y_valid)
	return accuracy

