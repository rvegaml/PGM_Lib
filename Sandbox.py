from __future__ import division
import numpy as np
from PGM_functions import Factor, Gibbs_sampling, transform_dataset, FactorProduct, MRF
from models import f_c_softmax_Group_LASSO
import tensorflow as tf

# Create a fake dataset
var_A = np.array([0])
card_A = np.array([2])
values_A = np.log(np.array([100, 50]))

var_B = np.array([1])
card_B = np.array([2])
values_B = np.log(np.array([50, 50]))

var_C = np.array([2])
card_C = np.array([3])
values_C = np.log(np.array([120,100,100]))

var_AB = np.array([0, 1])
card_AB = np.array([2, 2])
values_AB = np.log(np.array([100,120,120,120]))

var_AC = np.array([0, 2])
card_AC = np.array([2, 3])
values_AC = np.log(np.array([100,50,50,50,100,50]))

factor_A = Factor(var_A, card_A, values_A)
factor_B = Factor(var_B, card_B, values_B)
factor_C = Factor(var_C, card_C, values_C)
factor_AB = Factor(var_AB, card_AB, values_AB)
factor_AC = Factor(var_AC, card_AC, values_AC)

factor_list = [factor_A, factor_B, factor_C, factor_AB, factor_AC]

print('Creating dataset...')
dataset = Gibbs_sampling(factor_list, np.array([1,1,2]), 20000)

print(np.unique(dataset[:,0]))
print(np.unique(dataset[:,1]))
print(np.unique(dataset[:,2]))
print(np.min(dataset, axis=0))
print('Dataset created \n')

graph = MRF()
graph.train(dataset)
print(graph.adj_matrix)
print(graph.Z)
print(graph.compute_likelihood(np.array([[0,-1,0], [0,0,2]])))

factor_list = graph.factor_list
final_factor = factor_list[0]
for i in range(len(factor_list)-1):
	final_factor = FactorProduct(final_factor, factor_list[i+1])

print(final_factor.values / np.sum(final_factor.values))
prob =final_factor.values / np.sum(final_factor.values) 
print(prob[0] + prob[3])

# ----------------------------------------------------------------

# var_A = np.array([0])
# card_A = np.array([3])
# values_A = np.log(np.array([10, 10, 20]))

# var_B = np.array([1])
# card_B = np.array([3])
# values_B = np.log(np.array([5, 5, 5]))

# var_C = np.array([2])
# card_C = np.array([3])
# values_C = np.log(np.array([100,1,1]))

# var_D = np.array([3])
# card_D = np.array([3])
# values_D = np.log(np.array([50,10,30]))

# var_AC = np.array([0, 2])
# card_AC = np.array([3, 3])
# values_AC = np.log(np.array([100,10,10,10,100,10,1,1,100]))

# var_AD = np.array([0, 3])
# card_AD = np.array([3, 3])
# values_AD = np.log(np.array([10,50,50,50,10,50,50,50,10]))

# var_BD = np.array([1, 3])
# card_BD = np.array([3, 3])
# values_BD = np.log(np.array([30,1,30,30,1,30,30,1,100]))

# factor_A = Factor(var_A, card_A, values_A)
# factor_B = Factor(var_B, card_B, values_B)
# factor_C = Factor(var_C, card_C, values_C)
# factor_D = Factor(var_D, card_D, values_D)
# factor_AC = Factor(var_AC, card_AC, values_AC)
# factor_AD = Factor(var_AD, card_AD, values_AD)
# factor_BD = Factor(var_BD, card_BD, values_BD)

# factor_list = [factor_A, factor_B, factor_C, factor_D, factor_AC, factor_AD, factor_BD]

# print('Creating dataset...')
# dataset = Gibbs_sampling(factor_list, np.array([1,1,2,1]), 20000)
# print('Dataset created \n')

# ----------------------------------------------------------------
# num_variables = dataset.shape[1]
# pairwise_factors = []
# node_factors = []
# factor_list = []

# for x_id in range(num_variables):
# 	tf.reset_default_graph()
# 	new_dataset, Y, cardinality = transform_dataset(x_id, dataset)

# 	num_iter = 10000
# 	batch_size = 5000
# 	optimizer_name = tf.train.AdamOptimizer

# 	name = 'Test' 
# 	dim_classifier = np.array([new_dataset.shape[1], cardinality[x_id]])
# 	save_dir='./trained_variables_fc_H1_1000'

# 	F_C_Net = f_c_softmax_Group_LASSO(name, dim_classifier, cardinality, alpha=.01, save_dir=save_dir)
# 	F_C_Net.train(optimizer_name, new_dataset, Y, num_iter=num_iter, batch_size=batch_size)

# 	pairwise_factors.append(F_C_Net.final_weights)
# 	node_factors.append(F_C_Net.final_bias)

# # Create an empty adjacency matrix
# adj_matrix = np.zeros((num_variables, num_variables), dtype=np.int8)

# # Create the node and pairwise factor
# for x_id in range(num_variables):

# 	# Node factor
# 	new_Factor = Factor(np.array([x_id]), np.array([cardinality[x_id]]), np.exp(node_factors[x_id]))
# 	factor_list.append(new_Factor)

# 	rest_variables = []
# 	start_index = 0

# 	for i in range(num_variables):
# 		if i != x_id:
# 			rest_variables.append(i)

# 	for y_id in rest_variables:
# 		end_index = start_index + cardinality[y_id]

# 		if y_id > x_id:
# 			c_weights = pairwise_factors[x_id]
# 			c_factor_values_mat = c_weights[start_index:end_index, :]
# 			c_factor_values = np.reshape(c_factor_values_mat, (-1), order='F')

# 			# Pairwise factor
# 			if np.linalg.norm(c_factor_values) > 1E-6:
# 				adj_matrix[x_id, y_id] = 1
# 				adj_matrix[y_id, x_id] = 1
# 				new_Factor = Factor(np.array([x_id, y_id]), np.array([cardinality[x_id], cardinality[y_id]]), np.exp(c_factor_values))
# 				factor_list.append(new_Factor)
# 		start_index = end_index


# for factor in factor_list:
# 	print(factor.variables)
# 	print(factor.cardinality)
# 	print(factor.values)
# 	print('\n')

# print(adj_matrix)

# jointDistribution = Factor(factor_list[0].variables, factor_list[0].cardinality, factor_list[0].values)

# for i in range(len(factor_list)-1):
# 	jointDistribution = FactorProduct(jointDistribution, factor_list[i+1])

# Z = np.sum(jointDistribution.values)
# print(Z)

# var_A = np.array([0])
# card_A = np.array([3])
# values_A = np.array([10, 100, 20])

# var_B = np.array([10])
# card_B = np.array([2])
# values_B = np.array([5, 2])

# factor_A = Factor(var_A, card_A, values_A)
# factor_B = Factor(var_B, card_B, values_B)