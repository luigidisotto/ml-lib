import numpy as np
import csv
import pandas as pd

def retrieveSamplesFromCSV(filename, nx, ny, normalize='yes'):

   m = np.genfromtxt(filename, delimiter=",")
   X, Y = m[:,range(1,nx+1,1)], m[:,range(nx+1, nx+ny+1, 1)]

   #if normalize == 'yes':
   #	X_min, Y_min = X.min(axis=0), Y.min(axis=0)
   #	X = 0.1 + 0.8 * (X - X_min) / (X.max(axis=0) - X_min)
   #	Y = 0.1 + 0.8 * (Y - Y_min) / (Y.max(axis=0) - Y_min)

   #X = 1.0 * X
   #Y = 1.0 * Y

   return X, Y

def writeSamplesToCSV(filename, Y):
	 #df_x = pd.DataFrame(X)
	 df_y = pd.DataFrame(Y)
	 #df = pd.concat([df_x, df_y], axis=1)
	 df = pd.concat([df_y], axis=1)
	 df.to_csv(filename, index=True, header=False)

def retrieveSamplesFromMonk(filename):
	"""
		Number of Instances: 432

		Number of Attributes: 8 (including class attribute)

		Attribute information:
		    1. class: 0, 1 
		    2. a1:    1, 2, 3
		    3. a2:    1, 2, 3
		    4. a3:    1, 2
		    5. a4:    1, 2, 3
		    6. a5:    1, 2, 3, 4
		    7. a6:    1, 2
		    8. Id:    (A unique symbol for each instance)

		An example is the following
			class a1 a2 a3 a4 a5 a6 Id
	"""
	def onehot(nval, j):
		"""	
		The one-hot-encoding for ``nval`` objects.
		An array of lenght ``nval`` in one-hot-encoding representing the j-th object is returned.
		For example, with nval=2 and j=0 we return [.9, .1].
		"""
		h = np.full(nval, .1)
		h[j] = .9
		return h

	hot = { 'a1': 3,
		  	'a2': 3,
		  	'a3': 2,
		  	'a4': 3,
		  	'a5': 4,
		  	'a6': 2,
		  	'class': 2
		}

	# features indices inside of text line
	ax = {  'class': 0,
			'a1': 1,
	  	    'a2': 2,
	  	    'a3': 3,
	  	    'a4': 4,
	  	    'a5': 5,
	  	    'a6': 6
	  	  }

	attributes = ['a1', 'a2', 'a3', 'a4', 'a5', 'a6']

	samples = []

	with open(filename) as fp:
		X, Y = [], []
		for line in fp:
			xs = []
			l = line.strip().split(' ')
			# we read value ``v`` of attribute ``a`` and then take its one-hot-encoding,
			# namely, an array of lenght the size of objects represented by ``a`` in one-hot encoding.
			v = int(l[ax['class']])
			y = np.array([v]) #y = onehot(hot['class'], v)
			for a in attributes:
				aval = int(l[ax[a]])
				xs.append(onehot(hot[a], aval-1))
			x = np.concatenate(xs)
			xs = []
			X.append(x)
			Y.append(y)
		return np.array(X), np.array(Y)