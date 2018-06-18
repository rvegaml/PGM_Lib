from __future__ import division
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from models import f_c_softmax_Group_LASSO
import tensorflow as tf

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

		# Create the array that converts assignment to index
		temp = np.hstack([self.cardinality, 1])
		temp = np.flip(temp, axis=0)
		temp = np.flip(np.cumprod(temp), axis=0)
		
		self.convert_a_to_i = temp

		# Create a dictionary that maps every variable to an index
		var_to_ind = dict()
		for i in range(len(variables)):
			var_to_ind[variables[i]] = i

		self.var_to_ind = var_to_ind

	def index_to_assignment(self, index):
		# Extract the vector that contains the cumulative product of the cardinality
		temp = self.convert_a_to_i[1:]
		# Transform the vectors into matrices (This is needed to process several indexes
		# at the same time.)
		temp = np.matmul(np.ones([len(index), 1]), np.reshape(temp,[1,-1]))
		temp_index = np.matmul(np.reshape(index,[-1,1]), np.ones([1, temp.shape[1]]))
		temp_cardinality = np.matmul(np.ones([len(index), 1]), np.reshape(self.cardinality,[1,-1]))
		# Convert the index into the actual assignment
		temp = np.mod(np.floor(np.divide(temp_index,temp)), temp_cardinality)
		
		return np.int8(temp)
	
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

	def print_CPT(factor):
		'''
		Function that prints all the assignments ni the factor along with their probabilities
		'''
		assignments = factor.index_to_assignment(range(np.prod(factor.cardinality)))
		num_assignments, num_var = assignments.shape
		CPT = np.zeros((num_assignments, num_var+1))
		CPT[:, 0:num_var] = assignments
		CPT[:, -1] = factor.values
		
		print(CPT)

class MRF():
	def __init__(self, factor_list=[], adj_matrix=[]):
		self.factor_list = factor_list
		self.adj_matrix = adj_matrix

		if len(factor_list) == 0:
			self.Z = 0
		else:
			self.Z = compute_partition_function(factor_list)

	def compute_partition_function(self, factor_list):
		# Compute the partition function for the entire distribution
		jointDistribution = Factor(factor_list[0].variables, factor_list[0].cardinality, factor_list[0].values)

		for i in range(len(factor_list)-1):
			jointDistribution = FactorProduct(jointDistribution, factor_list[i+1])

		Z = np.sum(jointDistribution.values)

		return Z

	def train(self, dataset, alpha=.01):
		# Function to learn the potentials of a pairwise graph given a dataset and a regularization parameter alpha
		num_variables = dataset.shape[1]
		pairwise_factors = []
		node_factors = []
		factor_list = []

		for x_id in range(num_variables):
			tf.reset_default_graph()
			new_dataset, Y, cardinality = transform_dataset(x_id, dataset)

			num_iter = 10000
			batch_size = 5000
			optimizer_name = tf.train.AdamOptimizer

			name = 'Test' 
			dim_classifier = np.array([new_dataset.shape[1], cardinality[x_id]])
			save_dir='./trained_variables_fc_H1_1000'

			F_C_Net = f_c_softmax_Group_LASSO(name, dim_classifier, cardinality, alpha=.01, save_dir=save_dir)
			F_C_Net.train(optimizer_name, new_dataset, Y, num_iter=num_iter, batch_size=batch_size)

			pairwise_factors.append(F_C_Net.final_weights)
			node_factors.append(F_C_Net.final_bias)

		# Create an empty adjacency matrix
		self.adj_matrix = np.zeros((num_variables, num_variables), dtype=np.int8)

		# Create the node and pairwise factor
		for x_id in range(num_variables):

			# Node factor
			new_Factor = Factor(np.array([x_id]), np.array([cardinality[x_id]]), np.exp(node_factors[x_id]))
			factor_list.append(new_Factor)

			rest_variables = []
			start_index = 0

			for i in range(num_variables):
				if i != x_id:
					rest_variables.append(i)

			for y_id in rest_variables:
				end_index = start_index + cardinality[y_id]

				if y_id > x_id:
					c_weights = pairwise_factors[x_id]
					c_factor_values_mat = c_weights[start_index:end_index, :]
					c_factor_values = np.reshape(c_factor_values_mat, (-1), order='F')

					# Pairwise factor
					if np.linalg.norm(c_factor_values) > 1E-6:
						self.adj_matrix[x_id, y_id] = 1
						self.adj_matrix[y_id, x_id] = 1
						new_Factor = Factor(np.array([x_id, y_id]), np.array([cardinality[x_id], cardinality[y_id]]), np.exp(c_factor_values))
						factor_list.append(new_Factor)
				start_index = end_index

		self.factor_list = factor_list
		self.Z = self.compute_partition_function(factor_list)

	def compute_likelihood(self, dataset):
		'''
		Function that computes the likelihood of every instance in the dataset given a trained graph
		'''
		num_variables = dataset.shape[1]
		var_id = np.arange(num_variables)
		instance_factor_list = self.factor_list
		likelihood = np.zeros(dataset.shape[0])


		for instance_id in range(dataset.shape[0]):
			instance = dataset[instance_id, :]
			instance_factor_list = self.factor_list

			# Identify the variables whose values are missing
			missing_flag = (instance == -1)
			if np.any(missing_flag):
				var_to_marginalize = var_id[missing_flag]
			else:
				var_to_marginalize = []

			# Identify the factors that contain the variables to marginalize
			factors_to_marginalze = []
			other_factors = []

			if len(var_to_marginalize) != 0:
				for factor in self.factor_list:
					if np.any(np.in1d(var_to_marginalize, factor.variables)):
						factors_to_marginalze.append(factor)
					else:
						other_factors.append(factor)

				# Marginalize all the missing variables
				remaining_factors = []

				for var in var_to_marginalize:
					factors_with_c_variable = []

					for factor in factors_to_marginalze:
						if var in factor.variables:
							factors_with_c_variable.append(factor)
						else:
							remaining_factors.append(factor)

					# Multiply all the factor with the current variable
					new_factor = factors_with_c_variable[0]

					for i in range(len(factors_with_c_variable) - 1):
						new_factor = FactorProduct(new_factor, factors_with_c_variable[i+1])

					# Marginalize the current variable
					new_factor = FactorMarginalization(new_factor, var, np.sum)
					remaining_factors.append(new_factor)

					factors_to_marginalze = remaining_factors
					remaining_factors = []

				# The new factor list are the factors after marginalization
				instance_factor_list = factors_to_marginalze + other_factors


			unnorm_likelihood = 0
			for factor in instance_factor_list:
				c_val = factor.get_value_full_assignment(np.array([instance]))[0][0]
				unnorm_likelihood += np.log(c_val)

			likelihood[instance_id] = np.exp(unnorm_likelihood) / self.Z

		return likelihood


def transform_dataset(x_id, dataset):
	# Start by getting the cardinality of every variable on the dataset
	num_variables = dataset.shape[1]
	cardinality = np.zeros(num_variables, dtype=np.int16)
	features_var = []

	for var in range(num_variables):
		unique_values = np.unique(dataset[:, var])
		cardinality[var] = len(unique_values)

		if var != x_id:	
			features_var.append(var)

	print(cardinality)
	
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

def FactorProduct(factor_A, factor_B):
	# This function performs the factor product operation. The resulting factor has the entries in ascending order
	# of the variables id.

	if len(factor_A.variables) == 0:
		if len(factor_A.values) == 0:
			factor_C = Factor(factor_B.variables, factor_B.cardinality, factor_B.values)
		else:
			factor_C = Factor(factor_B.variables, factor_B.cardinality, factor_B.values*factor_A.values)

	if len(factor_B.variables) == 0:
		if len(factor_B.values) == 0:
			factor_C = Factor(factor_A.variables, factor_A.cardinality, factor_A.values)
		else:
			factor_C = Factor(factor_A.variables, factor_A.cardinality, factor_B.values*factor_A.values)

	# Set the variables present on the new factor
	var_C = np.union1d(factor_A.variables, factor_B.variables)

	# Identify the indexes of A and B in C
	map_A = np.zeros(len(factor_A.variables), dtype=np.int16)
	counter = 0
	for var in factor_A.variables:
		map_A[counter] = np.where(np.equal(var_C, var))[0][0]
		counter +=1

	map_B = np.zeros(len(factor_B.variables), dtype=np.int16)
	counter = 0
	for var in factor_B.variables:
		map_B[counter] = np.where(np.equal(var_C, var))[0][0]
		counter += 1

	# Set the cardinality of factor C
	card_C = np.zeros(len(var_C), dtype=np.int16)
	card_C[map_A] = factor_A.cardinality
	card_C[map_B] = factor_B.cardinality

	# Create the new factor C
	factor_C = Factor(var_C, card_C)

	# Fill the CPT
	assignments = factor_C.index_to_assignment(range(np.prod(factor_C.cardinality)))
	index_A = factor_A.assignment_to_index(assignments[:, map_A])
	index_B = factor_B.assignment_to_index(assignments[:, map_B])

	# To avoid underflow problems, make the multiplication in the log space
	new_prob = np.add(np.log(factor_A.values[index_A]), np.log(factor_B.values[index_B]))
	new_prob = np.exp(new_prob)

	factor_C.values = np.reshape(new_prob, (-1))

	return factor_C

def FactorMarginalization(factor, var_id, operation=np.sum):
	''' It marginalizes the var_id from the CPT using the operation defined in 'operation'.
	The function returns the unnormalized new factor without the variable to be marginalized.
	'''

	# Find the index of the variable to marginalize
	c_index = factor.var_to_ind[var_id]

	# Find the number of possible values that this variable might take
	c_card = factor.cardinality[c_index]

	# Create a new factor without the variable to marginalize
	new_variables = np.hstack([factor.variables[0:c_index], factor.variables[c_index+1:]])
	new_card = np.hstack([factor.cardinality[0:c_index], factor.cardinality[c_index+1:]])
	new_card = np.int16(new_card)
	new_factor = Factor(new_variables, new_card)

	# Find all the possible assignments of the new factor
	num_new_assignments = np.prod(new_card)
	new_possible_assignments = new_factor.index_to_assignment(range(num_new_assignments))

	# Find all the possible assignments without the variable to marginalize
	num_assignments = np.prod(factor.cardinality)
	possible_assignments = factor.index_to_assignment(range(num_assignments))
	possible_assignments = np.hstack([possible_assignments[:,0:c_index], possible_assignments[:,c_index+1:]])

	# Fill the new CPT
	for assignment in new_possible_assignments:
		i = new_factor.assignment_to_index(np.array([assignment]))
		prob_array = np.zeros(c_card)
		counter = 0
		for j in range(possible_assignments.shape[0]):
			line = possible_assignments[j]
			if np.array_equal(assignment, line):
				prob_array[counter] = factor.values[j]
				counter = counter + 1

		val = operation(prob_array)
		new_factor.values[i] = val

	return new_factor

def main():
	return -1

if __name__ == '__main__':
	main()