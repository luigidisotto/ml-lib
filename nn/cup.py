from NeuralNetwork import *
from HyperOptimizer import *
from plotting import *
from retrieve_dataset import *
import sys
from timeit import default_timer as timer

def main():

	# Learning the LOC dataset
	n_input, n_output = 10, 2

	X_tr, Y_tr = retrieveSamplesFromCSV("LOC-OSM2-TR.csv", n_input, n_output)

	X_blind, Y_blind = retrieveSamplesFromCSV("LOC-OSM2-TS.csv", n_input, 0)

	hyperspace = { 'learning_rate' : [.9],#[i/10 for i in range(1, 10)],
		           'momentum' : [.9],#[i/10 for i in range(1,10)],
		           'lambda' : [1e-05],#[10**(-i) for i in range(1,6)],
		           'topology': [[n_input, 32, n_output]]#[[n_input, 2**j, n_output] for j in range(1, 6)] # number of neurons are a power of 2
	             }

	#cup = NeuralNetwork(learning_rate=0.1, momentum = 0.1, lamda=0.0, layers=[10, 3, 2], loss=MeanEuclideanError)
	#cup_perf = cup.fit(X_tr, Y_tr)
	#plot_perf(cup_perf, 'perf_loc')
	start = timer()
	cup = HyperOptimizer(hyperspace, X_tr, Y_tr, loss=MeanSquaredError, eval_error=MeanEuclideanError,
									p=.75, k=5, scoring = ['valid_err', 'test_err'])
	cup.hoptimize()
	cup.blindtest(X_blind)
	end = timer()
	print('computing time = {} ms'.format(np.ceil((end-start)*1000)))
	sys.stdout.flush()

main()