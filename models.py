from __future__ import division
import numpy as np
import tensorflow as tf
from utils import fully_connected
from tensorflow.python.ops import variable_scope

class general_model():
	''' General model that contains some functions common to all the modules
	'''
	def __init__():
		self.X = []
		self.Y = []

	def train(self, optimizer_name, X, Y, num_iter=10000, batch_size=20, tol=1E-5, decay_steps=400, decay_rate=.96, starter_learning_rate=.1):


		# Get the number of samples of every hospital
		n_X = X.shape[0]

		# Create a random permutation of the indexes in the trainset of every hospital
		indexes_X = np.random.permutation(n_X)

		# Counter of the indexes that will be used for trainig
		start_X = 0

		# Set the parameters for the exponential decay in the learning rate
		global_step = tf.Variable(0, trainable=False)
		learning_rate = tf.train.exponential_decay(learning_rate=starter_learning_rate, global_step=global_step, 
			decay_steps=decay_steps, decay_rate=decay_rate, staircase=True)


		optimizer = optimizer_name(learning_rate=learning_rate)

		trainStep = optimizer.minimize(self.loss)

		threshold = 1E-2
		thresh_matrix = tf.cast(tf.greater(tf.abs(self.W_H1), threshold), tf.float32)
		new_weights = tf.multiply(self.W_H1, thresh_matrix)

		threshold_op = tf.assign(self.W_H1, new_weights)

		nan_matrix = tf.is_nan(self.W_H1)
		non_nan_weigths = tf.where(nan_matrix, tf.zeros(tf.shape(self.W_H1)), self.W_H1)
		nan_to_zero_op = tf.assign(self.W_H1, non_nan_weigths)

		saver = tf.train.Saver()

		# Train the model
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())

			prev_loss = np.inf

			for i in range(num_iter):
				# Get the data that will be used in this batch
				end_X = start_X + batch_size

				if end_X > n_X:
					start_X = 0
					end_X = batch_size
					indexes_X = np.random.permutation(n_X)

				c_indx_X = indexes_X[start_X:end_X]

				X_train = X[c_indx_X, :]
				Y_train = Y[c_indx_X, :]

				trainStep.run(feed_dict={self.X:X_train, self.Y:Y_train})
				sess.run(threshold_op)
				sess.run(nan_to_zero_op)

				if i % 100 == 0:
					c_loss = self.loss.eval(feed_dict={
						self.X:X, self.Y:Y
						})
					print('step %d, current loss %g' % (i, c_loss))

					train_acc = self.accuracy.eval(feed_dict={
						self.X:X, self.Y:Y
						})
					
					print('train accuracy %g' % (train_acc))

					if np.abs(c_loss - prev_loss) < tol:
						print('The difference in loss is lower than the specified tolerance')
						print('\n')
						break
					
					prev_loss = c_loss

				start_X = end_X
			# Save the model
			saver.save(sess, self.save_dir)
			self.final_weights = self.W_H1.eval()
			self.final_bias = self.b_H1.eval()

	def score(self, X=np.array([]), Y=np.array([])):
		saver = tf.train.Saver()

		with tf.Session() as sess:
			# print('Restoring the trained network')
			saver.restore(sess, self.save_dir)
			# for var in tf.global_variables():
			# 	print(var.name)
			# print('Model restored')

			# Check if we need to use the network of hospital 1 or 2
			n_X = X.shape[0]

			if n_X != 0:
				test_acc = self.accuracy.eval(feed_dict={
				self.X:X, self.Y:Y
				})
				print('Test accuracy %g' % (test_acc))

		return test_acc


class f_c_softmax_Group_LASSO(general_model):
	'''
	Class for a single layer, fully connected model.
	'''

	def __init__(self, name, dim_classifier, cardinality, alpha=10, save_dir='./trained_variables_fc'):
		dim_in_class = dim_classifier[0]
		dim_out_class = dim_classifier[1]

		self.save_dir = save_dir

		# Create the placeholders for the data of hospital 1 and hospital 2
		self.X = tf.placeholder(tf.float32, shape=[None, dim_in_class])
		self.Y = tf.placeholder(tf.float32, shape=[None, dim_out_class])

		# Create the network of the first transformer
		# Layer 1
		self.W_H1, self.b_H1, z_hat, y_hat = fully_connected(name, 'H_L1', self.X, dim_in_class,
			dim_out_class, tf.nn.softmax)

		# Compute the loss and accuracy
		ll_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.Y, logits=z_hat))

		# Compute the regularization term

		regularization = self.compute_regularization_term(self.W_H1, cardinality)

		self.loss = ll_loss + tf.multiply(tf.constant(alpha, dtype=tf.float32), regularization)
		
		correct_prediction_flag = tf.equal(tf.argmax(y_hat, 1), tf.argmax(self.Y,1))
		self.accuracy = tf.reduce_mean(tf.cast(correct_prediction_flag, tf.float32))

		self.prev_weights = tf.Variable(self.W_H1)

	def compute_regularization_term(self, W, cardinality):

		reg = 0
		start_index = 0

		for element in cardinality:
			end_index = start_index + element

			subset = W[start_index:end_index, :]

			reg+=tf.norm(subset)
			start_index = end_index

		return reg

def main():
	return -1

if __name__ == '__main__':
	main()