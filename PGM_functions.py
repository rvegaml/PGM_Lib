from __future__ import division
import numpy as np
from sklearn.preprocessing import OneHotEncoder

class Factor():
	def __init__(self, variables, cardinality, values=[]):
		'''
		This function initializes a factor. A factor is defined by three different arrays:
		- variables: Numpy array that contains the id of the variables in the scope of the factor. The id is a number
			between 0 and # variables.
		- cardinality: Numpy array that contains the cardinality of the variables in the scope of the factor.
			The order of the entries is the same than the order in variables.
		- values: Value of every assignment in the factor. It is a numpy array with prod(cardinality) entries,
			one per every combination of values.
		'''
		
		self.variables = variables
		self.cardinality = cardinality
		
		# Compute the number of entries that the factor should have:
		num_entries = np.int32(np.prod(self.cardinality))
		if len(values) > 0:
			if len(values) == num_entries:
				self.values = values
			else:
				print('Initializing factor with zeros')
				self.values = np.zeros(num_entries)
		else:
			print('Initializing factor with zeros')
			self.values = np.zeros(num_entries)
				
		# Create the one-hot encoding object
		self.onehot_encoder = OneHotEncoder(n_values=np.int16(num_entries), sparse=False)
		self.onehot_encoder.fit(np.reshape(range(num_entries), (-1,1)))
	
	def assignment_to_index(self, assignment):
		# Function that returns the index (in the values vector) of the given assignment.
		# Assignment is an array with len(self.variables) entries
		temp_card = np.concatenate([self.cardinality[1:], [1]])
		index = np.sum(temp_card*assignment, axis=1)
		
		return np.reshape(index, (-1,1))
	
	def full_assignment_to_index(self, x):
		# Function that returns the index (in the values vector) of the given assignment.
		# x is an array that contains the entire instance
		assignment = x[:,self.variables]
		return self.assignment_to_index(assignment)
	
	def assignment_to_one_hot_encoding(self, assignment):
		# Transform the current assignment to a one hot encoding.
		# Assignment is an array with len(self.variables) entries
		index = self.assignment_to_index(assignment)
		
		return self.onehot_encoder.transform(np.array(index))
		
	def full_assignment_to_one_hot_encoding(self, x):
		# Transform the current vector to a one hot encoding of the variables in the factor
		# x is an array that contains the entire instance
		assignment = x[:,self.variables]
		return self.assignment_to_one_hot_encoding(assignment)
	
	def get_value_assignment(self, assignment):
		index = self.assignment_to_index(assignment)
		index = np.reshape(index, (-1))
		
		return np.reshape(self.values[index], (-1,1))
	
	def get_value_full_assignment(self, x):
		assignment = x[:,self.variables]
		
		return self.get_value_assignment(assignment)

def transform_dataset(x_id, dataset):
	# Start by getting the cardinality of every variable on the dataset
	num_variables = dataset.shape[1]
	cardinality = np.zeros(num_variables, dtype=np.int16)
	features_var = []

	for var in range(num_variables):
		unique_values = np.unique(dataset[:, x_id])
		cardinality[var] = len(unique_values)

		if var != x_id:	
			features_var.append(var)
	
	# Determine the number of features in the vector
	features_card = cardinality[features_var]
	num_features = np.int16(np.sum(features_card))
	
	# Create the new dataset
	new_dataset = np.zeros((dataset.shape[0], num_features), dtype=np.int8)
	
	start_index = 0
	
	for i in range(len(features_card)):
		c_card = features_card[i]
		print(c_card)
		end_index = start_index + c_card
		
		# Create the one-hot encoding object
		onehot_encoder = OneHotEncoder(n_values=np.int16(c_card), sparse=False)
		onehot_encoder.fit(np.reshape(range(c_card), (-1,1)))

		new_dataset[:, start_index:end_index] = onehot_encoder.transform(np.reshape(dataset[:, features_var[i]], (-1,1)))
		start_index = end_index

	# Create the one-hot encoding object
	c_card = cardinality[x_id]
	onehot_encoder = OneHotEncoder(n_values=np.int16(c_card), sparse=False)
	onehot_encoder.fit(np.reshape(range(c_card), (-1,1)))
	Y = onehot_encoder.transform(np.reshape(dataset[:, x_id], (-1,1)))
	
	return new_dataset, np.int8(Y), cardinality

def prob_X_given_rest(x_id, evidence, factors_list):
	# Fucntion that computes the probability of X given the rest of the variables.
	# Inputs:
	# - x_id: Id from the variable whose probability will be computed
	# - evidence: Complete instance vector. The entry on x_id will be ignored.
	# - factors_list: List that contains all the factors in the graph.
	
	# Start by choosing the factors that contain x_id. The rest of the factor can be ignored
	relevant_factors = []
	
	for factor in factors_list:
		if x_id in factor.variables:
			relevant_factors.append(factor)
			
			# Identify the cardinality of X
			if len(factor.variables) == 1:
				X_card = factor.cardinality[0]
	
	# Create the different assignemnts X and an array to store their probabilities
	X_val = range(X_card)
	X_prob = np.zeros(X_card)
	
	X_assignments = np.dot(np.ones((X_card,1)), np.reshape(evidence, (1,-1)))
	X_assignments[:, x_id] = X_val
	X_assignments = np.int32(X_assignments)
	
	# Compute the unnormalized probability
	for factor in relevant_factors:
		temp_val = factor.get_value_full_assignment(X_assignments)
		X_prob += np.reshape(temp_val,(-1))
	
	X_prob = np.exp(X_prob)
	
	# Normalize the PMF
	X_prob = X_prob / np.sum(X_prob)
	
	return X_prob

def Gibbs_generate_next_sample(factor_list, sample):
	# Identify the number of variables
	num_var = len(sample)
	next_sample = np.array(sample)
	
	for x_id in range(num_var):
		prob_x_id = prob_X_given_rest(x_id, next_sample, factor_list)
		cardinality_x_id = len(prob_x_id)
		
		sample_x_id = np.random.choice(cardinality_x_id, p=prob_x_id)
		next_sample[x_id] = sample_x_id
	
	return np.int16(next_sample)

def Gibbs_sampling(factor_list, initial_sample, T):
	# Identify the number of variables
	num_var = len(initial_sample)
	
	# Generate a dataset with just zeros.
	dataset = np.zeros((T, num_var), dtype=np.int16)
	
	for i in range(T):
		new_sample = Gibbs_generate_next_sample(factor_list, initial_sample)
		dataset[i, :] = new_sample
	
	return dataset

def initializeFactors(dataset):
	# Function that creates the factors needed for the Group LASSO problem
	# Identify the number of variables and their respective cardinality
	num_variables = dataset.shape[1]
	cardinality = np.zeros(num_variables)
	factor_list = []
	
	for x_id in range(num_variables):
		# Create the node potentials
		unique_values = np.unique(dataset[:, x_id])
		cardinality[x_id] = len(unique_values)
		
		new_Factor = Factor(np.array([x_id]), np.array([cardinality[x_id]]))
		factor_list.append(new_Factor)
		
		# Create the pairwise potentials
		for y_id in range(num_variables):
			if x_id < y_id:
				# Create the node potentials
				unique_values = np.unique(dataset[:, y_id])
				cardinality[y_id] = len(unique_values)
				
				new_Factor = Factor(np.array([x_id, y_id]), np.array([cardinality[x_id], cardinality[y_id]]))
				factor_list.append(new_Factor)
	
	return factor_list