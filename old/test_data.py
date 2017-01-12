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


# Use a simple Gaussian Process Model to predict the acceleration of sensor 1
# Randomly samples a few rows to use as training set
# Test against the same number of randomly selected rows

PREDICT = 2 # Third entry in INDEX
INDEX = [1,0,4,2,9,6]
MAX_ROWS = 200*1000
SAMPLE_RATE = 200 # Hz
DOUBLE = np.dtype('d')

DATA_FILE = 'data.csv'
MODEL_FILE = 'models/svm-model-{0}'.format(INDEX[PREDICT])


def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

def select_random(matrix,n):
	# Select random rows from a matrix
	nrows = matrix.shape[0]
	index = np.random.randint(nrows,size=n)
	return matrix[index,:]

def sort_by(matrix,column):
	# Return a matrix sorted by the given column
	return matrix[matrix[:,column].argsort()]


# All acceleration is scaled such that the max accleration is 1.0
# This is difficult to keep consistent. Therefore we base our scales off
# the first 100k data points
def get_scales():
	print "Calculating column scales"
	ROWS = 100*1000
	data = genfromtxt(DATA_FILE, delimiter=',', max_rows=ROWS)
	mode_data = data[:,INDEX]
	mode_scales = mode_data.max(axis=0)
	return mode_scales
scales = get_scales()
 
# Read data from the excel spreadsheet
# Return three time series matrices
# @time. A column vector contaning the time [s]
# @x. A column matrix contaning acceleration from 6 sensors
# @y. A column vector containing acceleration from the remaining sensor
def read_data(start,end):
	with open(DATA_FILE,'r') as fd:
		for i in range(start):
			fd.next()
		time,x,y = read_next(fd, end-start)
	time = time+start/SAMPLE_RATE
	return (time,x,y)



# Read the next n points from a file
# More performant than read_data
def read_next(fd,n):
	print "Reading {0} rows from file".format(n)
	data = genfromtxt(fd, delimiter=',', max_rows=n)
	mode_data = data[:,INDEX]
	mode_data = mode_data/scales
	time = np.arange(0,n,dtype=DOUBLE).reshape((n, 1))/SAMPLE_RATE
	y = mode_data[:,PREDICT].reshape(n,1)
	x = np.delete(mode_data,PREDICT,axis=1)
	return time,x,y


# Train & test a model. Return the rmse
def predict():
	
	# Read in the data 
	cross_validate = False
	train_time,trainingX,trainingY = read_data(0,70000)
	test_time,testingX,testingY = read_data(130000,162000)

	# Choose model parameter search space
	parameters = {'kernel':['rbf'], 'gamma':linspace(0.0001,0.002,3), 'C':linspace(50,100,3)}
	print "Training with {0} rows".format(trainingX.shape[0])
	print "Training rows contain {0} features".format(trainingX.shape[1])

	if cross_validate:
		svr = SVR()
		clf = grid_search.GridSearchCV(svr, parameters, verbose=3)
		model = clf.fit(trainingX, trainingY)
		print model 
		print "Best params"
		print model.best_params_
	else:
		C = 75
		gamma = 0.0010499999999999999
		kernel = 'rbf'
		clf = SVR(C=C, kernel=kernel, gamma=gamma)
		model = clf.fit(trainingX, trainingY)

	# Make the prediction on the meshed x-axis (ask for MSE as well)
	print "Making predictions for {0} training rows".format(trainingX.shape[0])
	training_pred = model.predict(trainingX)
	print "Making predictions for {0} testing rows".format(testingX.shape[0])
	testing_pred = model.predict(testingX)

	# Plot original unscaled data
	scale = get_scales()[PREDICT]

	# Scatter of one acceleration against another
	fig = plt.figure()
	plt.plot(trainingX[:,2], scale*trainingY, 'b.', markersize=6, label=u'Observations')
	plt.plot(trainingX[:,2], scale*training_pred,  'rx', markersize=6, label=u'Predictions')
	plt.xlabel('Time [s]')
	plt.ylabel('Acceleration [g]')
	plt.legend(loc='upper left')

	# Plot the training time series
	fig = plt.figure()
	plt.plot(train_time, scale*trainingY, 'b', label=u'Observations')
	plt.plot(train_time, scale*training_pred, 'r', label=u'Predictions')
	plt.xlabel('Time [s]')
	plt.ylabel('Acceleration [g]')
	plt.title('Predictions for Training Set')

	# Plot the testing time series
	fig = plt.figure()
	plt.plot(test_time, scale*testingY, 'b', label=u'Observations')
	plt.plot(test_time, scale*testing_pred, 'r', label=u'Predictions')
	plt.xlabel('Time [s]')
	plt.ylabel('Acceleration [g]')
	plt.title('Acceleration Prediction for Testing Set')
	plt.legend(['Recorded values','Predicted Values'])
	plt.xlim([660,700])
	plt.savefig('test.png')

	print "Testing RMSE (%): "+`100*rmse(trainingY, training_pred)`
	print "Training RMSE (%): "+`100*rmse(testingY, testing_pred)`

	print "Writing model to pickle"
	with open(MODEL_FILE,'w') as fd:
		pickle.dump(model,fd)
	print "Finished writing pickle"
	return rmse(testingY, testing_pred)

if __name__=='__main__':
	predict()
	plt.show()

