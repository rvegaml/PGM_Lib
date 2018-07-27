'''
File: optimization_lib.py
Author: Roberto Vega
Email: rvega@ualberta.ca
Description:
	I will include some functions to perform Group Lasso
'''
import numpy as np

def softmax(Z):
	# Find the maximum element of every instance
	max_val = np.reshape(np.ndarray.max(Z, axis=1), (-1,1))
	exps = np.exp(Z - max_val)
	# Get the normalization constant o every instance
	sums = np.reshape(np.sum(exps, axis=1),(-1,1))
	
	return exps / sums

def group_lasso_proximal(w, reg_param, learning_rate):
	'''
	Implementation based on https://web.stanford.edu/~boyd/papers/pdf/prox_algs.pdf
	Section 6.5.4 Sum of norms
	'''
	norm_w = np.linalg.norm(w)
	temp = (1 - reg_param*learning_rate/norm_w)

	proximal = np.maximum(0, temp)*w

	return proximal

def gradient_l2_loss(w, XX, Xy):

	return np.dot(XX, w) - Xy

def gradient_cross_entropy_loss(W, X, y):
	grad = np.zeros(W.shape)
	num_samples = X.shape[0]
	
	Z = np.dot(X, W)
	prob = softmax(Z)
	
	temp = prob - y
	grad = np.dot(X.T, temp) / num_samples

	return grad

def GD_update_l2_group_lasso(w, XX, Xy, cardinality, learning_rate=.01, reg_param=0):
	# First compute the update ignoring the regularization term
	next_w_no_reg = w - learning_rate*gradient_l2_loss(w, XX, Xy)
	next_w = np.zeros(w.shape)
	next_w[0, :] = next_w_no_reg[0, :]
	# Then apply the proximal function to every group of weights
	# Do not apply regularization to the bias term, which is on the first column of X
	start_index = 1
	for card in cardinality:
		end_index = start_index + card
		# Extract the weights of every group
		c_weights = next_w_no_reg[start_index:end_index,:]
		c_proximal = group_lasso_proximal(c_weights, reg_param*np.sqrt(card), learning_rate)
		next_w[start_index:end_index,:] = c_proximal

		start_index = end_index

	return next_w

def GD_update_cross_entropy_group_lasso(w, X, y, cardinality, learning_rate=.01, reg_param=0):
	# First compute the update ignoring the regularization term
	next_w_no_reg = w - learning_rate*gradient_cross_entropy_loss(w, X, y)
	next_w = np.zeros(w.shape)
	next_w[0, :] = next_w_no_reg[0, :]
	# Then apply the proximal function to every group of weights
	# Do not apply regularization to the bias term, which is on the first column of X
	start_index = 1
	for card in cardinality:
		end_index = start_index + card
		# Extract the weights of every group
		c_weights = next_w_no_reg[start_index:end_index, :]
		c_proximal = group_lasso_proximal(c_weights, reg_param*np.sqrt(card), learning_rate)
		next_w[start_index:end_index,:] = c_proximal

		start_index = end_index

	return next_w

def compute_group_lasso_penalty(w, cardinality, reg_param):
	# Compute the regularization error
	start_index = 1
	reg_loss = 0

	for card in cardinality:
		end_index = start_index + card
		# Extract the weights of every group
		c_weights = w[start_index:end_index, :]
		reg_loss += (np.linalg.norm(c_weights) * np.sqrt(card))

		start_index = end_index

	return  reg_param*reg_loss


def compute_l2_group_lasso_cost(w, X, y, cardinality, reg_param=0):
	# Compute the MSE
	y_hat = np.dot(X, w)

	# Compute the mean squared error
	num_samples = X.shape[0]
	error = y - y_hat
	MSE = 0.5 * np.dot(error.T, error) / num_samples
	
	# Compute the regularization error
	reg_loss = compute_group_lasso_penalty(w, cardinality, reg_param)

	loss = MSE + reg_loss

	return loss

def compute_cross_entropy_group_lasso_cost(W, X, y, cardinality, reg_param=0):
	# Compute the output
	Z = np.dot(X, W)
	prob = softmax(Z)
	
	num_samples = X.shape[0]
	
	log_likelihood = -np.log(np.sum(np.multiply(prob, y), axis=1))
	
	cross_entropy_loss = np.sum(log_likelihood) / num_samples
	
	reg_loss = compute_group_lasso_penalty(W, cardinality, reg_param)
	
	loss = cross_entropy_loss + reg_loss
	
	return loss


def line_search(w, X, y, XX, Xy, cost_function, update_function, cardinality=[], 
	learning_rate_max=1, tau=0.7, tol = 1E-5, max_iter=300, reg_param=0):

	c_learning_rate = learning_rate_max
	c_w = w
	objective = cost_function(c_w, X, y, cardinality, reg_param)

	improved_flag = False

	for i in range(max_iter):
		if update_function == GD_update_l2_group_lasso:
			new_w = update_function(c_w, XX, Xy, cardinality, c_learning_rate, reg_param)
		elif update_function == GD_update_cross_entropy_group_lasso:
			new_w = update_function(c_w, X, y, cardinality, c_learning_rate, reg_param)

		new_cost = cost_function(new_w, X, y, cardinality, reg_param)

		if new_cost < (objective - tol):
			improved_flag = True
			break

		c_learning_rate = c_learning_rate*tau

	if improved_flag == False:
		new_w = w
		c_learning_rate = 0


	return new_w, c_learning_rate

def linear_regression_group_lasso(X, y, cardinality, reg_param, max_iter=10000, tol=1E-5):
	# Add a column of 1's to the dataset
	num_samples = X.shape[0]
	new_X = np.hstack([np.ones((num_samples, 1)), X])

	# Compute useful matrices here to avid doing it on the loop.
	XX = np.dot(new_X.T, new_X) / num_samples
	Xy = np.dot(new_X.T, y) / num_samples

	c_learning_rate = 10

	# Generate the initial weight
	old_w = np.ones((new_X.shape[1],1))

	for i in range(max_iter):
		new_w, c_learning_rate = line_search(old_w, new_X, y, XX, Xy, compute_l2_group_lasso_cost,
			GD_update_l2_group_lasso, cardinality, reg_param=reg_param, learning_rate_max=c_learning_rate, tol=tol)

		if c_learning_rate == 0:
			break
		else:
			old_w = new_w

	print('Error')
	error = y - np.dot(new_X, new_w)
	print(np.mean(error))
	print('Num iter: {0:d}'.format(i))

	return new_w

class Linear_Regression_Group_LASSO():
	def __init__(self, cardinality, reg_param):
		self.W = []
		self.cardinality = cardinality
		self.reg_param = reg_param

	def train(self, X, y, max_iter=10000, tol=1E-5):
		w = linear_regression_group_lasso(X, y, self.cardinality, self.reg_param, max_iter, tol)
		self.W = w

	def predict(self, X):
		# Append a column of ones for the bias
		num_samples = X.shape[0]
		new_X = np.hstack([np.ones((num_samples, 1)), X])
		# Make the predictions
		predictions = np.dot(new_X, self.W)

		return predictions

	def compute_variance_residuals(self, X, Y):
		predictions = self.predict(X)
		residuals = np.reshape(Y, (-1)) - np.reshape(predictions, (-1))

		return np.var(residuals)


def multiclass_logistic_regression_group_lasso(X, y, cardinality, reg_param, max_iter=10000, tol=1E-5):
	# Add a column of 1's to the dataset
	num_samples, num_features = X.shape
	num_classes = y.shape[1]

	new_X = np.hstack([np.ones((num_samples, 1)), X])

	# Compute useful matrices here to avid doing it on the loop.
	XX = np.dot(new_X.T, new_X) / num_samples
	Xy = np.dot(new_X.T, y) / num_samples

	c_learning_rate = 10

	# Generate the initial weight
	old_w = np.zeros((num_features+1, num_classes))

	for i in range(max_iter):
		new_w, c_learning_rate = line_search(old_w, new_X, y, XX, Xy, compute_cross_entropy_group_lasso_cost,
			GD_update_cross_entropy_group_lasso, cardinality, reg_param=reg_param, learning_rate_max=c_learning_rate, tol=tol)

		if c_learning_rate == 0:
			break
		else:
			old_w = new_w

	print('Error')
	cross_entropy_loss = compute_cross_entropy_group_lasso_cost(
		new_w, new_X, y, cardinality, reg_param=0)
	print(cross_entropy_loss)
	print('Num iter: {0:d}'.format(i))

	return new_w

class Multiclass_Logistic_Regression_Group_LASSO():
	def __init__(self, cardinality, reg_param):
		self.W = []
		self.cardinality = cardinality
		self.reg_param = reg_param

	def train(self, X, y, max_iter=10000, tol=1E-5):
		w = multiclass_logistic_regression_group_lasso(X, y, self.cardinality, self.reg_param, max_iter, tol=tol)
		self.W = w

	def predict(self, X):
		# Append a column of ones for the bias
		num_samples = X.shape[0]
		new_X = np.hstack([np.ones((num_samples, 1)), X])
		# Make the predictions
		predictions = np.argmax(softmax(np.dot(new_X, self.W)), axis=1)

		return predictions

def main():
	return -1

if __name__ == '__main__':
	main()