import numpy as np
from NeuralNetwork import *
from HyperOptimizer import *
from plotting import *
from retrieve_dataset import *
import sys
from timeit import default_timer as timer

def main():

	# Learning the XOR
	# (obs.: sensitive to standard dev. we use to init. weights)
	#X = np.array([[.1, .1], [.1, .9], [.9, .1], [.9, .9]], dtype=float)
	#Y = np.array([[.1], [.9], [.9], [.1]], dtype=float)
	#xor = NeuralNetwork(learning_rate= 1.0, layers=[2, 2, 1], loss=MeanSquaredError, sigma=0.05)
	#perf_xor = xor.fit(X, Y)
	#plot_perf(perf_xor, 'perf_xor')
	

	# --- MONK Benchmarks ---

	n_input, n_output = 17, 1

	hyperspace = { 'learning_rate' : [i/10 for i in range(1, 10)],
		           'momentum' : [i/10 for i in range(1,10)],
		           'lambda' : [0.0],#[10**(-i) for i in range(1,6)],
		           'topology': [[n_input, 2**j, n_output] for j in range(1, 6)]
	             }

	# Monk-1

	X_train, Y_train = retrieveSamplesFromMonk("../data/monks-1.train")

	X_test, Y_test = retrieveSamplesFromMonk("../data/monks-1.test")

	"""
	start = timer()
	monk1 = HyperOptimizer(hyperspace, X_train, Y_train, MeanSquaredError, p=.70, k=5, 
												scoring=['valid_err', 'test_err', 'accuracy'], task='classification', 
												X_ts=X_test, Y_ts=Y_test)
	monk1.hoptimize()
	end = timer()
	print('monk1 computing time = {} ms'.format(np.ceil((end-start)*1000)))
	sys.stdout.flush()
	"""

	monk1 = NeuralNetwork(learning_rate=0.6, momentum=0.8, lamda=1e-5, max_epochs=5000,
							layers=[17, 8, 1], loss=MeanSquaredError, task='classification')
	perf_monk1 = monk1.fit(X_train, Y_train)
	mse = monk1.score(X_test, Y_test)
	print('test error = {}'.format(mse))
	plot_perf(perf_monk1, 'perf_monk1')
	plot_accuracy(perf_monk1, 'acc_monk1')

	# Monk-2
	
	"""
	X_train, Y_train = retrieveSamplesFromMonk("monks-2.train")

	X_test, Y_test = retrieveSamplesFromMonk("monks-2.test")

	monk2 = HyperOptimizer(hyperspace, X_train, Y_train, MeanSquaredError, p=.70, k=5,
										task = 'classification', scoring = ['valid_err', 'accuracy', 'test_err'], 
										X_ts=X_test, Y_ts = Y_test)
	start = timer()
	monk2.hoptimize()
	end = timer()
	print('monk2 computing time = {} ms'.format(np.ceil((end-start)*1000)))
	sys.stdout.flush()
	"""
	
	#monk2 = NeuralNetwork(learning_rate=0.1, momentum=0.3, lamda = 0.0, max_epochs=50000,
	#						layers=[17, 3, 1], loss=MeanSquaredError, task='classification')
	#perf_monk2 = monk2.fit(X_train, Y_train)
	#mse = monk2.mse(X_valid, Y_valid)
	#print('test error = {}'.format(mse))
	#plot_perf(perf_monk2, 'perf_monk2')
	#plot_accuracy(perf_monk2, 'acc_monk2')

	
	# Monk-3
	"""
	X_train, Y_train = retrieveSamplesFromMonk("monks-3.train")

	X_test, Y_test = retrieveSamplesFromMonk("monks-3.test")

	monk3 = HyperOptimizer(hyperspace, X_train, Y_train, MeanSquaredError, p=.70, k=5,
										task = 'classification', scoring = ['valid_err', 'accuracy', 'test_err'], 
										X_ts=X_test, Y_ts = Y_test)

	start = timer()
	monk3.hoptimize()
	end = timer()

	print('monk3 computing time = {} ms'.format(np.ceil((end-start)*1000)))
	sys.stdout.flush()
	"""

	#monk3 = NeuralNetwork(learning_rate=0.1, momentum = 0.5, lamda = 0.0, max_epochs=50000,
	#						layers=[17, 4, 1], loss=MeanSquaredError, task='classification')
	#perf_monk3 = monk3.fit(X_train, Y_train)
	#mse = monk3.mse(X_valid, Y_valid)
	#print('test error = {}'.format(mse))
	#plot_perf(perf_monk3, 'perf_monk3')
	#plot_accuracy(perf_monk3, 'acc_monk3')
	

main()