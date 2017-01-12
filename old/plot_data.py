import pickle
import time as timer
import numpy as np
from numpy import linspace
from numpy import genfromtxt
from sklearn import grid_search
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from matplotlib import pyplot as plt
from scipy.signal import detrend
from sklearn.svm import SVR
from test_data import rmse,sort_by,get_scales,read_data

# Use a simple Gaussian Process Model to predict the acceleration of sensor 1
# Randomly samples a few rows to use as training set
# Test against the same number of randomly selected rows

PREDICT = 2 # Third entry in INDEX
INDEX = [1,0,4,2,9,6]
MAX_ROWS = 200*1000
SAMPLE_RATE = 200 # Hz
DOUBLE = np.dtype('d')
BLUE = '#0098DB'

scales = get_scales()
scale = get_scales()[PREDICT]


def plot():
	train_time,trainingX,trainingY = read_data(0,70000)
	test_time,testingX,testingY = read_data(130000,162000)

	plt.plot(trainingX[:,2], scale*trainingY, 'b.', markersize=6, label=u'Observations', color=BLUE)
	plt.xlabel('Time [s]')
	plt.ylabel('Acceleration [g]')
	plt.legend(loc='upper left')

	# Plot the training time series
	fig = plt.figure()
	plt.plot(train_time, scale*trainingY, 'b', label=u'Observations', color=BLUE)
	plt.xlabel('Time [s]')
	plt.ylabel('Acceleration [g]')
	plt.title('Predictions for Training Set')

	# Plot the testing time series
	fig = plt.figure()
	plt.plot(test_time, scale*testingY, 'b', label=u'Observations', color=BLUE)
	plt.xlabel('Time [s]')
	plt.ylabel('Acceleration [g]')
	plt.title('Bridge Acceleration')
	plt.legend(['Recorded time series','Predicted Values'])
	plt.xlim([660,700])
	plt.savefig('test.png')



if __name__=='__main__':
	plot()
	plt.show()

