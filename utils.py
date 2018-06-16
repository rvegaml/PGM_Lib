'''
File: utils.py
Author:	Roberto Vega
Description:
	It contains helper functions used for the training of the neural networks
'''

import tensorflow as tf
import numpy as np
from tensorflow.python.ops import variable_scope

def fully_connected(name, label, x, dim_in, dim_out, transfer, reuse=False):
	with variable_scope.variable_scope(name, reuse=reuse):
		with variable_scope.variable_scope(label, reuse=reuse):
				W = variable_scope.get_variable('W', [dim_in, dim_out])
				b = variable_scope.get_variable('b', [dim_out])

	z_hat = tf.matmul(x, W) + b
	y_hat = transfer(z_hat)

	return W, b, z_hat, y_hat

def main():
	return -1

if __name__ == '__main__':
	main()