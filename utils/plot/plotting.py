import matplotlib.pyplot as plt
plt.switch_backend('agg')
from matplotlib import rc
import numpy as np

def plot_perf(perf, title):
	plt.style.use('ggplot')
	plt.figure(figsize=(15,5))
	plt.ylabel("Error")
	plt.xlabel("Epochs")
	#plt.ylim((0, 1))
	plt.plot(perf["tr_err_hist"], '-', linewidth=4.0, label="mee on Train-set")
	#plt.plot(perf["loss_hist"], '.', linewidth=4.0, label="mse loss on Train-set")
	plt.plot(perf["va_err_hist"], '3', linewidth=4.0, label="mee on Validation-set")
	plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1), ncol=2)
	plt.savefig(title+'.pdf')
	plt.clf()
	

def plot_accuracy(perf, title):
	plt.style.use('ggplot')
	plt.figure(figsize=(15,5))
	plt.ylabel('Accuracy')
	plt.xlabel('Epochs')
	plt.plot(perf["tr_acc_hist"], label="Train-set")
	plt.plot(perf["va_acc_hist"], label="Validation-set")
	plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1), ncol=2)
	plt.savefig(title+'.pdf')
	plt.clf()

def scatter_plot(title, Y_ts):
	plt.style.use('ggplot')
	#plt.rc('text', usetex=True) # doesn't work on machines lacking tex
	plt.xlabel('y1')
	plt.ylabel('y2')
	N = Y_ts.shape[0]
	colors = np.random.rand(N)
	area = np.pi* (15 * np.random.rand(N))**2  # 0 to 15 point radii
	#plt.scatter(Y_tr[:,0], Y_tr[:,1], alpha=0.5, label="Train-set")
	plt.scatter(Y_ts[:,0], Y_ts[:,1], alpha=0.5, label="Train-set", s=area, c=colors)
	#cs=121
	#plt.scatter(Y_ts[:,0], Y_ts[:,1], color='r', s=2*s, marker='^', alpha=.4, label="Blind-set")
	plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1), ncol=2)
	plt.savefig(title + '.pdf')
	plt.clf()

def scattering(title, lbl, Y):
	plt.style.use('ggplot')
	#plt.rc('text', usetex=True)
	plt.xlabel(r'y_1')
	plt.ylabel(r'y_2')
	N = Y.shape[0]
	colors = np.random.rand(N)
	area = np.pi* (15 * np.random.rand(N))**2  # 0 to 15 point radii
	plt.scatter(Y[:,0], Y[:,1], s=area, c=colors, alpha=0.5, label=lbl)
	plt.legend(loc='upper left', bbox_to_anchor=(0.5, 1), ncol=2)
	plt.savefig(title + '.pdf')
	plt.clf()
