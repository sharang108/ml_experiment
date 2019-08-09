import os
import csv 
import numpy as np

# def load_dataset(file, file_lines, features):	
def load_dataset(file):
	with open(file, 'r') as work_file:
		reader = list(csv.reader(work_file))
		total = len(reader)
		train_set = reader[:round(total * 0.8)]
		val_set = reader[:round(total * 0.2)]
		features = len(train_set[0][:8])
		x_train = np.zeros((len(train_set), features))
		y_train = np.zeros((len(train_set), 1))
		x_val = np.zeros((len(val_set), features))
		y_val = np.zeros((len(val_set), 1))
		
		for index, val in enumerate(train_set):
			x_train[index] = val[:features]
			y_train[index] = val[-1]

		for index, val in enumerate(val_set):
			x_val[index] = val[:features]
			y_val[index] = val[-1]

	return x_train, y_train, x_val, y_val

def activation(fun, var):
	val = 0.0
	if fun == 'tanh':
		val = np.tanh(var)
		# val = np.exp(2 * var) - 1 / np.exp(2 * var) + 1
	
	elif fun == 'sigmoid':
		val = 1/ (1 + np.exp(-var))

	elif fun == 'relu':
		val = max(0, var)
	
	elif fun == 'softmax':
		pass

	return val

def loss_calc(y, a):
	return -(np.dot(y, np.log(a)) + np.dot((1-y), np.log(a)))
	# return -(y * np.log(a) + (1-y) * np.log(a))

x_train, y_train, x_val, y_val = load_dataset('workwith_data.csv')
# Weights inititaed in trasponsed shape
# 0.001 is the ideal weights multiplier else log loss goes nan due to log 0 or -ve
# basically log 0 converges to inf and log -x is imaginary number so nan
w1 = np.random.randn(x_train.shape[1], 3) * 0.001
w2 = np.random.randn(3, 1) * 0.001
# baises over layers
b1 = 0.0
b2 = 0.0
cost = 0.0
dw1 = 0.0
db1 = 0.0
dw2 = 0.0
db2 = 0.0
samples = x_train.shape[0]
lr = 0.01
for i in range(100):
	# forward pass
	z1 = np.dot(x_train, w1) + b1
	a1 = activation(fun='tanh', var=z1)
	z2 = np.dot(a1, w2) + b2
	a2 = activation(fun='sigmoid', var=z2)
	loss = loss_calc(y_train.T, a2)
	cost =  np.sum(loss)/samples
	# Backprop
	dz2 = a2 - y_train
	dw2 += np.dot(dz2.T, a1)/samples
	db2 += dz2/samples
	tanh_diff = 1 - np.square(z1)
	dz1 = (w2.T * dz2) * tanh_diff
	dw1 += np.dot(dz1.T, x_train)/samples
	db1 += dz1/samples
	w1 = w1 - lr * dw1
	w2 = w2 - lr * dw2
	print('iteration ' + str(i) + ' cost'+str(cost))




