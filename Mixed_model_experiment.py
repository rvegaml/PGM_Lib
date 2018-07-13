import numpy as np
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from PGM_functions import Mixed_MRF
from sklearn.preprocessing import StandardScaler

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

# Get the Z-score of the data
scaler = StandardScaler()
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
y = np.reshape(y, (-1))

# Divide the data into train and test
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=11879)

# Separate the data into the groups SCZ and HC
# Train the models
HC = Y_train==0
SCZ = Y_train==1

graph_HC = Mixed_MRF()
graph_HC.train(X_train[HC, :], cardinality, weights, reg_param=.01)

graph_SCZ = Mixed_MRF()
graph_SCZ.train(X_train[SCZ, :], cardinality, weights, reg_param=.01)

likelihood_1 = graph_1.compute_mixed_loglikelihood(X_test)
likelihood_2 = graph_2.compute_mixed_loglikelihood(X_test)

Y_pred = (likelihood_2 > likelihood_1) + 0

acc = np.sum(Y_pred == Y_test) / len(Y_test)
print(acc)

print(len(Y_test))