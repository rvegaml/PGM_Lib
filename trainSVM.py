'''
Function:	trainSVM(dataset, labels)
Description:
	This function will train a SVM model given a labeled dataset and a set of parameters.
Inputs:
	- dataset: numpy array of shape (num_samples, num_features)
		. Every row is an instance and the columns are the features.
	- labels: numpy array of shape (num_samples,) that contains the class of every 
		instance in dataset.
	- params: dictionary that contains the following values:
		+ kernel: RBF, Linear
		+ C: vector that contains the values of C to use
		+ gamma: vector that contains the values of gamma to use
		+ fSel: Feature selection method: None
		+ k: Number of folds to use in internal cross validation
		+ classWeights: dictionary with the weights for every class.Default = 1
		+ performance: Metric to optimize: F1 | accuracy
'''
from __future__ import division
import pandas as pd
import numpy as np

def trainSVM(dataset, labels, params):

	# Import the requires libraries
	from sklearn.model_selection import StratifiedKFold
	from sklearn import svm
	from sklearn.metrics import f1_score

	# Create the k-fold object
	skf = StratifiedKFold(n_splits=params['k'])
	counter = 0

	# Check if we are using class weights, if not, they are all set to 1
	if 'classWeights' in params:
		class_weights = params['classWeights']
	else:
		unique_labels = np.unique(labels)
		class_weights = dict()
		for lab in unique_labels:
			class_weights[lab] = 1

	# Check what should be optimized
	if 'performance' in params:
		performance = params['performance']
	else:
		performance = 'accuracy'

	# Determine the max number of iterations
	if 'max_iter' in params:
		max_iter = params['max_iter']
	else:
		max_iter = 10000

	# Choose if you want to solve the primal or dual form in the 
	# linear SVM
	if 'dual' in params:
		dual = params['dual']
	else:
		dual = True

	if 'penalty' in params:
		penalty = params['penalty']
	else:
		penalty = 'l2'


	# Select the best parameters for the RBF kernel
	if params['kernel'] == 'RBF':
		best_c = 0
		best_gamma = 0
		best_per = -1

		num_exp = len(params['C']) * len(params['gamma'])
		

		for c in params['C']:
			for gamma in params['gamma']:
				counter = counter + 1
				print("Experiment {0:d} out of {1:d}".format(counter, num_exp))
				print('Trying C = {0:.3f}, and gamma = {1:.3f}'.format(c, gamma))
				current_per = np.zeros(params['k'])
				fold = 0

				# Estimate the accuracy of the given parameters
				for train_indx, test_indx in skf.split(dataset, labels):
					X_train, X_test = dataset[train_indx,:], dataset[test_indx,:]
					y_train, y_test = labels[train_indx], labels[test_indx]

					# Train a model
					clf = svm.SVC(C=c, kernel='rbf', gamma=gamma, class_weight=class_weights, 
						max_iter=max_iter)
					clf.fit(X_train, y_train)
					predictions = clf.predict(X_test)

					# Test the model
					if performance == 'accuracy':
						current_per[fold] = clf.score(X_test, y_test)
					elif performance == 'F1':
						current_per[fold] = f1_score(y_test, predictions)

					fold = fold + 1

				mean_per = np.mean(current_per)

				if mean_per > best_per:
					best_per = mean_per
					best_c = c
					best_gamma = gamma

		# Train the final model with the best parameters
		clf = svm.SVC(C=best_c, kernel='rbf', gamma=best_gamma, class_weight=class_weights,
			max_iter=max_iter)
		clf.fit(dataset, labels)

	# Select the best parameters for the Linear kernel
	elif params['kernel'] == 'linear':
		best_c = 0
		best_per = -1
		num_exp = len(params['C'])

		for c in params['C']:
			counter = counter + 1
			print("Experiment {0:d} out of {1:d}".format(counter, num_exp))
			print("Trying C = {0:.3f}".format(c))
			current_per = np.zeros(params['k'])
			fold = 0

			# Estimate the accuracy of the given parameters
			for train_indx, test_indx in skf.split(dataset, labels):
				print('\t Fold:{0}'.format(fold))
				X_train, X_test = dataset[train_indx,:], dataset[test_indx,:]
				y_train, y_test = labels[train_indx], labels[test_indx]

				# Train a model
				clf = svm.LinearSVC(C=c, class_weight=class_weights, 
					max_iter=max_iter, dual=dual, penalty=penalty)
				clf.fit(X_train, y_train)
				predictions = clf.predict(X_test)

				# Test the model
				if performance == 'accuracy':
					current_per[fold] = clf.score(X_test, y_test)
				elif performance == 'F1':
					current_per[fold] = f1_score(y_test, predictions)
				
				fold = fold + 1

			mean_per = np.mean(current_per)
			print(mean_per)
			if mean_per > best_per:
				best_per = mean_per
				best_c = c

		# Train the final model with the best parameters
		clf = svm.LinearSVC(C=best_c, class_weight=class_weights,
			max_iter=max_iter, dual=dual, penalty=penalty)
		clf.fit(dataset, labels)

	return clf

def main():
	return -1

if __name__ == '__main__':
	main()