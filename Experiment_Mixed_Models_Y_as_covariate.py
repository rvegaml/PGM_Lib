# Import the required modules
import numpy as np
from sklearn.model_selection import StratifiedKFold
from PGM_functions import Mixed_MRF_2
from sklearn.preprocessing import StandardScaler
from trainSVM import trainSVM
from PGM_functions import prob_X_given_rest
import pickle
import pandas as pd

# Load the ReHo data
avg_signal_covaried_dict = pickle.load(open('./obj_szReho_0.01-0.08_avg_signal_covaried_dict.pkl', 'rb'))
# Print the different atlases available
print(avg_signal_covaried_dict.keys())

# Select a particular atlas
harvard_cort_25 = avg_signal_covaried_dict['harvard_cort_25']
# Extract the number of continuous variables
num_cont = harvard_cort_25.shape[1]

# Extract the ID of the data
identifier_dict = pickle.load(open('./obj_szReho_0.01-0.08_identifier_dict.pkl', 'rb'))
identifier_dict.keys()

# MAke the data zero mean
scaler = StandardScaler(with_std=False, with_mean=True)
norm_data = scaler.fit_transform(harvard_cort_25)

# Import the genomic data
Genedata_dict = dict()
Diagnosis_dict = dict()

dataset = pd.read_csv('./HiSC_Genotypes_PGM_enum.csv')
num_discrete = dataset.shape[1] - 1

for i in range(dataset.shape[0]):
	Subject_ID = dataset['Group_Code'][i]
	Genedata_dict[Subject_ID] = dataset.iloc[i,1:].values
	
	if Subject_ID[0] == 'C':
		Diagnosis_dict[Subject_ID] = 0
	else:
		Diagnosis_dict[Subject_ID] = 1

# Create a dictionary with the continuous data
Group_code = identifier_dict['Group_Code']
Brain_data = dict()

for i in range(len(Group_code)):
	subject_id = Group_code[i]
	data = norm_data[i, :]
	Brain_data[subject_id] = data

# Create the dataset that combines brain and gene data
X = np.zeros((0, num_discrete + num_cont))
y = np.zeros((0,1))

for subject_id in Diagnosis_dict.keys():
	if subject_id in Genedata_dict.keys() and subject_id in Brain_data.keys():
		c_vector = np.concatenate([Genedata_dict[subject_id], Brain_data[subject_id]])
		X = np.vstack([X, c_vector])
		y = np.vstack([y, np.array([Diagnosis_dict[subject_id]])])

cardinality = np.int16(np.concatenate([np.ones(num_discrete)*3, np.ones(num_cont)]))
weights = np.sqrt(cardinality)
y = np.reshape(y, (-1))


# -----------------------------------------------------------------
# Train the models
# -----------------------------------------------------------------
n_splits = 5

skf = StratifiedKFold(n_splits=n_splits)
iteration = 0
acc_train = np.zeros(n_splits)
acc_test = np.zeros(n_splits)

for train_index, test_index in skf.split(X, y):
	X_train, X_test = X[train_index,:], X[test_index,:]
	y_train, y_test = y[train_index], y[test_index]

	# Build a new model including the Y as part of the graphical model
	all_train = np.hstack([np.reshape(y_train, (-1,1)), X_train])
	all_card = np.concatenate([[2], cardinality])

	# Train the models
	graph = Mixed_MRF_2(all_card)
	graph.train(all_train, reg_param=.01, num_iter=20000)


	# Make predictions on the train and test set
	num_instances = all_train.shape[0]
	predictions = np.zeros(num_instances)
	var = 0

	for i in range(num_instances):
		c_instance = np.array([all_train[i, :]])
		
		prob = prob_X_given_rest(var, c_instance, graph.cardinality, 
			continuous_id=graph.continuous_id, J=graph.J, 
			alpha=graph.alpha, discrete_factors=graph.discrete_factors,
			mixed_factors=graph.mixed_factors)
		
		if prob[0][1] > prob[0][0]:
			predictions[i] = 1

	acc_train[iteration] = np.sum(predictions == all_train[:, 0])/num_instances

	all_test = np.hstack([np.reshape(Y_test, (-1,1)), X_test])
	num_instances = all_test.shape[0]
	predictions = np.zeros(num_instances)
	var = 0

	for i in range(num_instances):
		c_instance = np.array([all_test[i, :]])
		
		prob = prob_X_given_rest(var, c_instance, graph.cardinality, 
			continuous_id=graph.continuous_id, J=graph.J, 
			alpha=graph.alpha, discrete_factors=graph.discrete_factors,
			mixed_factors=graph.mixed_factors)
		
		if prob[0][1] > prob[0][0]:
			predictions[i] = 1

	acc_test[iteration] = np.sum(predictions == all_test[:, 0])/num_instances

	print('Iteration: {0:d}'.format(iteration))
	print('Train accuracy: {0:f}'.format(acc_train))
	print('Test accuracy: {0:f}'.format(acc_test))
	print('\n')

print('Cross validation train: {0:f}'.format(np.mean(acc_train)))
print('Cross validation test: {0:f}'.format(np.mean(acc_test)))