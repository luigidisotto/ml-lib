import random
import multiprocessing as mp
import sys
import logging

import pandas as pd
import numpy as np

from .MLP import MLP
from .loss import MeanSquaredError
from utils.plot import scatter_plot, plot_accuracy, plot_perf

logger = logging.getLogger(__name__)


def to_csv(filename, y):
	 df_y = pd.DataFrame(y)
	 df = pd.concat([df_y], axis=1)
	 df.to_csv(filename, index=True, header=False)

class HyperOptimizer:
	"""  k-cross model selection/validation  """
	def __init__(
		self, 
		hyperspace, 
		X_tr, 
		Y_tr, 
		loss=MeanSquaredError, 
		eval_error=MeanSquaredError, 
		p=0.75, 
		k=5, 
		scoring=['valid_err'],
		task='regression',
		X_ts=None,
		Y_ts=None
	):
		""" 
			hyperspace: a dictionary P -> [1,...,n] mapping from the set of the hyper-parameters to a list of values
			X_tr, Y_tr: the set of examples onto which we would perform training during model selection
			loss: the specific loss we would use
			p: 0 <= p <= 1, fraction of ``Trset`` to be used for training, and the remaining (1-p) fraction to be used as test set
			k: the ``k``-fold order
			task: if the problem is of 'classification' or 'regression'
			scoring: a list of scoring metric to be used in risk estimation, for example ['mse', 'accuracy'] if we are trying to
					 to solve a 'classification' task and we want to find both the model with lowest accuracy and lowest mse (though
					 could also coincide)
			X_ts, Y_ts: the previously unseen examples with which we would test model generalization capability, otherwise if not
						available, we made it by taking a fraction (1-p) of the training examples.

		"""
		self._hyperspace   = hyperspace
		self._loss         = loss
		self._eval         = eval_error
		self._scoring      = scoring
		self._k            = k
		self._p            = p
		self.bestmodel     = []
		self._risks        = []
		self._task         = task

		self.modelFactory(self._hyperspace)
		self.split_dataset(X_tr, Y_tr, X_ts, Y_ts)


	def split_dataset(self, X_tr, Y_tr, X_ts, Y_ts):
		n = X_tr.shape[0]
		idx_tr = random.sample(range(n), int(self._p*n)) if X_ts is None else [] # random dataset samples going to be the train./valid. set
		idx_ts = [i for i in range(n) if i not in idx_tr] if X_ts is None else [] # remaining dataset samples building up the test set

		self._X_tr  	   = (X_tr if X_ts is not None else X_tr[idx_tr,:])
		self._Y_tr         = (Y_tr if Y_ts is not None else Y_tr[idx_tr,:])
		self._X_ts         = (X_ts if X_ts is not None else X_tr[idx_ts,:])
		self._Y_ts         = (Y_ts if Y_ts is not None else Y_tr[idx_ts,:])


	def modelFactory(self, H):
		"""  
			H: the hyperparameters' space to be explored
			return: a list of models for every hyparameters combination
		"""
		self._models = []
		for eta in H['learning_rate']:
			for alpha in H['momentum']:
				for lamda in H['lambda']:
					for topology in H['topology']:
						self._models.append(MLP(learning_rate=eta, momentum = alpha, \
													 		lamda=lamda, layers=topology, loss=self._loss, eval_error=self._eval,
													 			 task=self._task))

	def riskEstimation(self, X, Y, model):
		""" k-cross risk estimation """
		n = X.shape[0]
		n_fold = int(n/self._k)
		ts, vs, accs = [], [], []

		sys.stdout.flush()
		str = f'[risk estimation]: model=(eta = {model._lrate}, momentum = {model._momentum}, lambda = { model._lamda}, nn = {model.n_hidden()})'
		logger.info(str)
		sys.stdout.flush()
		for f in range(0, n, n_fold):
			m = f+n_fold if f+n_fold < n else n
			fold_va = [idx for idx in range(f, m)] # validation fold
			fold_tr = [idx for idx in range(0, n) if idx not in fold_va] # training folds
			perf = model.fit(X.take(fold_tr, axis=0), Y.take(fold_tr, axis=0))
			err = model.score(X.take(fold_va, axis=0), Y.take(fold_va, axis=0))
			for score in self._scoring:
				if score == 'accuracy':
					accs.append(perf['va_acc'])
				if score == 'valid_err':
					vs.append(perf['va_err'])
				if score == 'test_err':
					ts.append(err)

		return { 'valid_err' : np.mean(vs)   if vs else [],
			     'test_err'  : np.mean(ts)   if ts else [],
				 'accuracy'  : np.mean(accs) if accs else []
				}
			
	def partiallyAppliedRiskEstimation(self, model):
		"""
			partial function application to be used in paralell map
		"""
		return self.riskEstimation(self._X_tr, self._Y_tr, model)

	def modelSelection(self):
		#self._risks = [self.riskEstimation(self._X_tr, self._Y_tr, model) \
		#												for model in self._models]
		nw = mp.cpu_count()
		p = mp.Pool(processes=nw)
		self._risks = p.map(self.partiallyAppliedRiskEstimation, self._models)
		for score in self._scoring:
			if score == 'valid_err':
				# model giving lowest validation error
				m = min([(self._risks[j]['valid_err'], self._models[j]) \
												for j in range(len(self._models))], key=lambda t: t[0])
				self.bestmodel.append((m[1], 'valid_err'))
			if score == 'test_err':
				# model giving lowest test error
				m = min([(self._risks[j]['test_err'], self._models[j]) \
												for j in range(len(self._models))], key=lambda t: t[0])
				self.bestmodel.append((m[1], 'test_err'))
			if score == 'accuracy':
				# model giving the highest accuracy
				m = max([(self._risks[j]['accuracy'], self._models[j]) \
														for j in range(len(self._models))], key=lambda t: t[0])
				self.bestmodel.append((m[1], 'accuracy'))
		


	def modelAssessment(self):
		for m in self.bestmodel:
			model = m[0]
			score = m[1]
			str = f'model=(eta = {model._lrate}, momentum = {model._momentum}, lambda = { model._lamda}, nn = {model.n_hidden()})'
			model.init_free_vars()
			# retrain on entire training set, then test learning abilities on unseen examples
			perf = model.fit(self._X_tr, self._Y_tr)
			
			if self._task == 'classification':
				logger.info(
					'[model assessment]: ' + str + '--' + score + ', tr = {}, vl = {}, loss = {}, tr. acc. = {}, va. acc = {}'.format(perf['tr_err'], 
					perf['va_err'],
					perf['loss'],
					perf['tr_acc'],
					perf['va_acc'])
					)
			else:
				logger.info(
					'[model assessment]: ' + str + '--' + score + ', tr = {}, vl = {}, loss = {}'.format(perf['tr_err'],
					perf['va_err'],
					perf['loss'])
					)

			sys.stdout.flush()

			err = model.score(self._X_ts, self._Y_ts)

			if self._task == 'regression':
				logger.info('[test error]: ' + str + '--' + score + ', ts = {err}')
			else:
				Y_pred = model.predict(self._X_ts)
				acc = model.accuracy(self._Y_ts, (Y_pred >= 0.5).astype(int))
				logger.info('[test error]: ' + str + '--' + score + ', ts = {err}, acc = {acc}')
				plot_accuracy(perf, str + '--' + score + '-ACCURACY')

			sys.stdout.flush()

			plot_perf(perf, str+'--'+score)
			
	
	def blindtest(self, X_blind):
		for m in self.bestmodel:
			model = m[0]
			score = m[1]
			str = f'model=(eta = {model._lrate}, momentum = {model._momentum}, lambda = { model._lamda}, nn = {model.n_hidden()})'
			logger.info('[blind test]: ' + str)
			sys.stdout.flush()
			Y_blind = model.predict(X_blind)
			to_csv('blind-test-'+str+'-'+score, y=Y_blind)
			scatter_plot('scatter-plot'+str+'--'+score, Y_blind)


	def hoptimize(self):
		self.modelSelection()
		self.modelAssessment()