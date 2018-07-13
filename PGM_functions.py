from __future__ import division
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from models import Linear_Regression_Group_LASSO, Multiclass_Logistic_Regression_Group_LASSO
from models import f_c_softmax_Group_LASSO
from scipy.stats import multivariate_normal
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
				# print('Initializing factor with zeros')
				self.values = np.zeros(num_entries)
		else:
			# print('Initializing factor with zeros')
			self.values = np.zeros(num_entries)
				
		# Create the one-hot encoding object
		# self.onehot_encoder = OneHotEncoder(n_values=np.int16(num_entries), sparse=False)
		# self.onehot_encoder.fit(np.reshape(range(num_entries), (-1,1)))

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
	
	# def assignment_to_one_hot_encoding(self, assignment):
	# 	# Transform the current assignment to a one hot encoding.
	# 	# Assignment is an array with len(self.variables) entries
	# 	index = self.assignment_to_index(assignment)
		
	# 	return self.onehot_encoder.transform(np.array(index))
		
	# def full_assignment_to_one_hot_encoding(self, x):
	# 	# Transform the current vector to a one hot encoding of the variables in the factor
	# 	# x is an array that contains the entire instance
	# 	assignment = x[:,self.variables]
	# 	return self.assignment_to_one_hot_encoding(assignment)
	
	def get_value_assignment(self, assignment):
		index = self.assignment_to_index(assignment)
		index = np.reshape(index, (-1))
		
		return np.reshape(self.values[index], (-1,1))
	
	def get_value_full_assignment(self, x, cardinality):
		continuous_card = (cardinality == 1)
		temp_x = np.array(x)
		temp_x[:, continuous_card] = 0
		assignment = np.int16(temp_x[:,self.variables])
		
		return self.get_value_assignment(assignment)

	def print_CPT(factor):
		'''
		Function that prints all the assignments ni the factor along with their probabilities
		'''
		assignments = factor.index_to_assignment(list(range(np.prod(factor.cardinality))))
		num_assignments, num_var = assignments.shape
		CPT = np.zeros((num_assignments, num_var+1))
		CPT[:, 0:num_var] = assignments
		CPT[:, -1] = factor.values
		
		print(CPT)

class Mixed_MRF():
	'''
		Class to learn and compute the likelihood of data combining discrente and continuous variables.
	'''
	def __init__(self, cardinality, discrete_factors=[], mixed_factors=[], J=[], alpha=[]):
		'''
			- discrete_factors is a list of pairwise factors for discrete variables
			- mixed_factors is a list of pairwise factors between a discrete and a continuous variable
			- J inverse of the covariance matrix of the continuous variables
			- mu: vector with the mean value of the continuous variables
		'''

		# Identify the ID of the discrete and continuous variables
		self.identify_continuous_discrete(cardinality)

		self.discrete_factors = discrete_factors
		self.mixed_factors = mixed_factors
		self.J = J
		self.alpha = alpha
		self.cardinality = cardinality

		# Compute the partition function of the discrete variables
		if len(discrete_factors) == 0:
			self.Z = 0
		else:
			self.Z = self.compute_partition_function(discrete_factors)

	def compute_partition_function(self, factor_list):
		# Compute the partition function for the entire distribution
		jointDistribution = Factor(factor_list[0].variables, factor_list[0].cardinality, factor_list[0].values)

		for i in range(len(factor_list)-1):
			jointDistribution = FactorProduct(jointDistribution, factor_list[i+1])

		Z = np.sum(jointDistribution.values)

		return Z

	def train(self, dataset, weights, reg_param=.01, num_iter=100000,
		batch_size=500,starter_learning_rate=.001, tol=1E-3):

		# Function to learn the potentials of a pairwise graph given a dataset and a regularization parameter alpha
		num_variables = dataset.shape[1]
		pairwise_discrete_weights = []
		node_discrete_weights = []
		continuous_weights = []
		continuous_bias = []
		discrete_factors = []
		mixed_factors = []
		cardinality = self.cardinality

		# Only weights above the connection tolerance will be taken
		connection_tolerance = 1E-3

		# Detect the number of features that are continuous
		continuous_flag = (cardinality == 1)
		num_continuous = np.sum(continuous_flag)

		J = np.zeros((num_continuous, num_continuous))
		alpha = np.zeros(num_continuous)

		continuous_counter = 0

		all_variables = list(range(num_variables))

		# Create the linear or multiclass logistic regression for each of the variables
		for x_id in all_variables:
			print('Variable {0:d} out of {1:d}'.format(x_id, len(all_variables)))
			tf.reset_default_graph()
			new_dataset, Y, features_card = transform_dataset(x_id, dataset, cardinality)

			features_weight = np.concatenate([weights[0:x_id], weights[x_id+1:]])

			optimizer_name = tf.train.AdamOptimizer

			name = 'Regression'
			c_card = cardinality[x_id]
			dim_classifier = np.array([new_dataset.shape[1], c_card])
			save_dir='./trained_variables_fc_H1_1000'

			if c_card == 1:
				F_C_Net = Linear_Regression_Group_LASSO(
					name, dim_classifier, features_card, features_weight, reg_param, save_dir=save_dir)
				F_C_Net.train(optimizer_name, new_dataset, Y, num_iter=num_iter, batch_size=batch_size, 
					starter_learning_rate=starter_learning_rate, tol=tol)
				var_residuals = F_C_Net.compute_variance_residuals(new_dataset, Y)

				J[continuous_counter, continuous_counter] = 1/var_residuals

				continuous_counter +=1

				continuous_weights.append(F_C_Net.final_weights/var_residuals)
				continuous_bias.append(F_C_Net.final_bias/var_residuals)
			else:

				F_C_Net = Multiclass_Logistic_Regression_Group_LASSO(
					name, dim_classifier, features_card, features_weight, reg_param, save_dir=save_dir)
				F_C_Net.train(optimizer_name, new_dataset, Y, num_iter=num_iter, batch_size=batch_size,
					starter_learning_rate=starter_learning_rate, tol=tol)

				print(F_C_Net.final_weights)

				pairwise_discrete_weights.append(F_C_Net.final_weights)
				node_discrete_weights.append(F_C_Net.final_bias)

		# Create an empty adjacency matrix
		self.adj_matrix = np.zeros((num_variables, num_variables), dtype=np.int8)

		# Create the Factors and the inverse covariance matrix
		discrete_counter = 0
		continuous_counter = 0

		for x_id in range(num_variables):

			c_card = cardinality[x_id]

			if c_card == 1:

				alpha[continuous_counter] = continuous_bias[continuous_counter]

				rest_variables = all_variables[0:x_id] + all_variables[x_id+1:]

				start_index = 0
				next_cont = continuous_counter + 1

				for y_id in rest_variables:
					y_card = cardinality[y_id]
					end_index = start_index + y_card

					if y_id > x_id:
						c_weights = continuous_weights[continuous_counter]
						c_factor_values_mat = c_weights[start_index:end_index, :]
						c_factor_values = np.reshape(c_factor_values_mat, (-1), order='F')

						# Pairwise factor
						if np.linalg.norm(c_factor_values) > connection_tolerance:
							self.adj_matrix[x_id, y_id] = 1
							self.adj_matrix[y_id, x_id] = 1

							# Identify if this is a discrete or mixed factor.
							if y_card == 1:
								J[continuous_counter, next_cont] = c_factor_values
								J[next_cont, continuous_counter] = c_factor_values
								next_cont +=1
							else:
								new_Factor = Factor(np.array([x_id, y_id]), 
									np.array([cardinality[x_id], cardinality[y_id]]), 
									np.exp(c_factor_values))
								mixed_factors.append(new_Factor)
					start_index = end_index


				continuous_counter += 1

			else:
				# Node factor
				new_Factor = Factor(np.array([x_id]), np.array([cardinality[x_id]]), np.exp(node_discrete_weights[discrete_counter]))
				discrete_factors.append(new_Factor)

				rest_variables = all_variables[0:x_id] + all_variables[x_id+1:]

				start_index = 0

				for y_id in rest_variables:
					y_card = cardinality[y_id]
					end_index = start_index + y_card

					if y_id > x_id:
						c_weights = pairwise_discrete_weights[discrete_counter]
						c_factor_values_mat = c_weights[start_index:end_index, :]
						c_factor_values = np.reshape(c_factor_values_mat, (-1), order='F')

						print(c_factor_values)
						print(np.linalg.norm(c_factor_values))

						# Pairwise factor
						if np.linalg.norm(c_factor_values) > connection_tolerance:
							self.adj_matrix[x_id, y_id] = 1
							self.adj_matrix[y_id, x_id] = 1
							new_Factor = Factor(np.array([x_id, y_id]), 
								np.array([cardinality[x_id], cardinality[y_id]]), 
								np.exp(c_factor_values))
							# Identify if this is a discrete or mixed factor.
							if y_card == 1:
								mixed_factors.append(new_Factor)
							else:
								discrete_factors.append(new_Factor)
					start_index = end_index

				discrete_counter += 1


		self.discrete_factors = discrete_factors
		self.mixed_factors = mixed_factors
		self.J = J
		self.alpha = alpha

		if (len(discrete_factors) != 0) or (len(mixed_factors) != 0):
			self.Z = self.compute_partition_function(discrete_factors)


	def identify_continuous_discrete(self, cardinality):
		self.continuous_id = []
		self.discrete_id = []

		num_variables = len(cardinality)

		for var in range(num_variables):
			if cardinality[var] == 1:
				self.continuous_id.append(var)
			else:
				self.discrete_id.append(var)

		self.continuous_id = np.array(self.continuous_id)
		self.discrete_id = np.array(self.discrete_id)

	def compute_discrete_likelihood(self, dataset):
		'''
		Function that computes the likelihood of every instance in the dataset given a trained graph
		'''
		cardinality_discrete = self.cardinality[self.discrete_id]

		var_id = self.discrete_id
		instance_factor_list = self.discrete_factors
		likelihood = np.zeros(dataset.shape[0])


		for instance_id in range(dataset.shape[0]):
			
			instance = dataset[instance_id, :]

			# Check that the instance has correct values (not greater than cardinality)
			error_value_flag = np.greater_equal(instance[self.discrete_id], cardinality_discrete)
			if np.any(error_value_flag):
				likelihood[instance_id] = 1E-10
			else:

				instance_factor_list = self.discrete_factors

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
					for factor in self.discrete_factors:
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
					c_val = factor.get_value_full_assignment(
						np.array([instance]), self.cardinality)[0][0]
					unnorm_likelihood += np.log(c_val)

				likelihood[instance_id] = np.exp(unnorm_likelihood) / self.Z

		return likelihood

	def compute_mixed_loglikelihood(self, dataset):
		Sigma = np.linalg.inv(self.J)

		num_variables = dataset.shape[1]
		num_discrete = len(self.discrete_id)
		num_continuous = len(self.continuous_id)

		# Compute the likelihood of the discrete variables
		discrete_likelihood = self.compute_discrete_likelihood(dataset)

		# Now go to the mixed model
		# Create a matrix that will contain all the rho vectors
		rho_matrix = np.zeros((num_discrete, num_continuous))
		mixed_likelihood = np.zeros(dataset.shape[0])

		for i in range(dataset.shape[0]):
			instance = dataset[i,:]

			for factor in self.mixed_factors:
				if factor.cardinality[0] == 1:
					cont_var = factor.variables[0]
					disc_var = factor.variables[1]
				else:
					cont_var = factor.variables[1]
					disc_var= factor.variables[0]

				cont_indx = np.where(self.continuous_id == cont_var)[0][0]
				disc_indx = np.where(self.discrete_id == disc_var)[0][0]

				rho_matrix[disc_indx, cont_indx] = factor.get_value_full_assignment(
					np.array([instance]), self.cardinality)[0][0]

			rho_vector = np.sum(rho_matrix, axis=0)
			temp = self.alpha + rho_vector
			x = instance[self.continuous_id]

			c_mu = np.dot(Sigma, temp)
			likelihood = multivariate_normal.pdf(x, c_mu, Sigma)

			if likelihood < 1E-250:
				mixed_likelihood[i] = 1E-250
			else:
				mixed_likelihood[i] = likelihood

		return np.log(discrete_likelihood) + np.log(mixed_likelihood)



class Discrete_MRF():
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

	def train(self, dataset, cardinality, alpha=.01):
		# Function to learn the potentials of a pairwise graph given a dataset and a regularization parameter alpha
		num_variables = dataset.shape[1]
		pairwise_factors = []
		node_factors = []
		factor_list = []

		for x_id in range(num_variables):
			tf.reset_default_graph()
			new_dataset, Y, features_card = transform_dataset(x_id, dataset, cardinality)

			num_iter = 10000
			batch_size = dataset.shape[0]
			batch_size = 5000
			optimizer_name = tf.train.AdamOptimizer

			name = 'Test' 
			dim_classifier = np.array([new_dataset.shape[1], cardinality[x_id]])
			save_dir='./trained_variables_fc_H1_1000'

			F_C_Net = f_c_softmax_Group_LASSO(name, dim_classifier, features_card, alpha, save_dir=save_dir)
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
		self.cardinality = cardinality

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

			# Check that the instance has correct values (not greater than cardinality)
			error_value_flag = np.greater_equal(instance, self.cardinality)
			if np.any(error_value_flag):
				likelihood[instance_id] = 1E-10
			else:

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

def transform_dataset(x_id, dataset, cardinality):
	# x_id is a scalar indicating the column number of the variable to predict
	# dataset is a numpy array of n x p
	# cardinality is a numpy vector. The cardinality for continuous values is 1

	# Identify the ID of the variables that will be used as features
	num_variables = dataset.shape[1]
	var_id = list(range(num_variables))
	features_var = var_id[0:x_id] + var_id[x_id+1:]

	# Determine the number of features in the vector
	features_card = cardinality[features_var]
	num_features = np.int16(np.sum(features_card))
	
	# Create the new dataset
	new_dataset = np.zeros((dataset.shape[0], num_features))
	
	start_index = 0
	
	for i in range(len(features_card)):
		c_card = features_card[i]
		end_index = start_index + c_card
		
		if c_card == 1:
			new_dataset[:, start_index:end_index] = np.reshape(
				dataset[:, features_var[i]], (-1,1))
		else:
			# Create the one-hot encoding object
			onehot_encoder = OneHotEncoder(n_values=np.int16(c_card), sparse=False)
			onehot_encoder.fit(np.reshape(list(range(c_card)), (-1,1)))

			new_dataset[:, start_index:end_index] = onehot_encoder.transform(
				np.reshape(dataset[:, features_var[i]], (-1,1)))

		start_index = end_index

	# Create the one-hot encoding object
	c_card = cardinality[x_id]
	if c_card == 1:
		Y = np.reshape(dataset[:, x_id], (-1,1))
	else:
		onehot_encoder = OneHotEncoder(n_values=np.int16(c_card), sparse=False)
		onehot_encoder.fit(np.reshape(list(range(c_card)), (-1,1)))
		Y = onehot_encoder.transform(np.reshape(dataset[:, x_id], (-1,1)))
	
	return new_dataset, Y, features_card

def prob_X_given_rest(x_id, evidence, cardinality, continuous_id=[], 
	discrete_factors=[], mixed_factors=[], J=[], alpha=[]):
	# Fucntion that computes the probability of X given the rest of the variables.
	# Inputs:
	# - x_id: Id from the variable whose probability will be computed
	# - evidence: Complete instance vector. The entry on x_id will be ignored.
	# - continuous_id is a numpy array that contains the indexes of the variables that are continuous
	# - factors_list: List that contains all the factors in the graph.

	card_x_id = cardinality[x_id]

	if card_x_id == 1:
		# Identify the continuous_id of x_id
		cont_id = np.where(continuous_id == x_id)[0][0]

		# Identify the number of continuous variables
		num_continuous = len(continuous_id)
		range_continuous_variables = np.arange(num_continuous)

		# Extract the elements to compute the probabiluty
		other_cont_id_range = np.concatenate([range_continuous_variables[0:cont_id], range_continuous_variables[cont_id + 1:]])
		other_cont_id_general = np.concatenate([continuous_id[0:cont_id], continuous_id[cont_id + 1:]])

		c_alpha = alpha[cont_id]
		B_ss = J[cont_id, cont_id]
		w = J[cont_id, other_cont_id_range]
		x_cont = evidence[:, other_cont_id_general]

		# Extract the mixed factors
		ro_mixed_factors = 0
	
		for factor in mixed_factors:
			if x_id in factor.variables:
				ro_mixed_factors += factor.get_value_full_assignment(evidence, cardinality)[0][0]

		unnormalized_mu = ro_mixed_factors + c_alpha + np.dot(x_cont, np.reshape(w, (-1, 1)))

		mu = unnormalized_mu/B_ss
		sigma_sqrd = 1/B_ss

		parameters = [mu[0][0], sigma_sqrd]

				
	else:
		# Create the different assignemnts X and an array to store their probabilities
		X_val = list(range(card_x_id))
		X_prob = np.zeros(card_x_id)
		
		X_assignments = np.dot(np.ones((card_x_id,1)), np.reshape(evidence, (1,-1)))
		X_assignments[:, x_id] = X_val

		# Start by choosing the factors that contain x_id. The rest of the factor can be ignored
		for factor in discrete_factors:
			if x_id in factor.variables:
				temp_val = factor.get_value_full_assignment(X_assignments, cardinality)
				X_prob += np.reshape(temp_val,(-1))

		for factor in mixed_factors:
			if x_id in factor.variables:
				# Identify the other variable
				other_var = np.setdiff1d(factor.variables, x_id)[0]
				temp_val = factor.get_value_full_assignment(X_assignments, cardinality)
				X_prob += np.reshape(temp_val,(-1))*evidence[0, other_var]
		
		X_prob = np.exp(X_prob)
		
		# Normalize the PMF
		X_prob = X_prob / np.sum(X_prob)

		parameters = [X_prob]
	
	return parameters

def Gibbs_generate_next_sample(sample, cardinality, continuous_id=[], 
	discrete_factors=[], mixed_factors=[], J=[], alpha=[]):
	# Identify the number of variables
	num_var = sample.shape[1]
	next_sample = np.array(sample, dtype=np.float32)
	
	for x_id in range(num_var):
		params = prob_X_given_rest(x_id, next_sample, cardinality, 
			continuous_id, discrete_factors, mixed_factors, J, alpha)

		cardinality_x_id = cardinality[x_id]
		
		if cardinality_x_id == 1:
			mu, sigma_sqrd = params
			sample_x_id = np.random.normal(mu, np.sqrt(sigma_sqrd))
		else:
			prob_x_id = params[0]
			sample_x_id = np.random.choice(cardinality_x_id, p=prob_x_id)

		next_sample[0, x_id] = sample_x_id
	
	return next_sample


def Gibbs_sampling(initial_sample, T, cardinality, continuous_id=[], 
	discrete_factors=[], mixed_factors=[], J=[], alpha=[]):
	# Identify the number of variables
	num_var = initial_sample.shape[1]
	
	# Generate a dataset with just zeros.
	dataset = np.zeros((T, num_var))
	
	for i in range(T):
		new_sample = Gibbs_generate_next_sample(initial_sample, cardinality, continuous_id,
			discrete_factors, mixed_factors, J, alpha)
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
	assignments = factor_C.index_to_assignment(list(range(np.prod(factor_C.cardinality))))
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
	new_possible_assignments = new_factor.index_to_assignment(list(range(num_new_assignments)))

	# Find all the possible assignments without the variable to marginalize
	num_assignments = np.prod(factor.cardinality)
	possible_assignments = factor.index_to_assignment(list(range(num_assignments)))
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