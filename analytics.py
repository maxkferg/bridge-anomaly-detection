"""
An analysis class for the bridge sensors

Usage:
time,train = read_data(0,70000)
time,test = read_data(130000,162000)

sensors = BridgeSensors()
sensors.train(train)
sensors.test(test)
sensors.error(test)
"""
import utils
import numpy as np
from utils import SCALES
from numpy import linspace
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from matplotlib import pyplot as plt

class BridgeSensors:
	"""A class that predicts whether data from the bridge sensors 
	matches the data that this model was trained on.

	Internally we train a model to predict the acceleration at each sensor
	We then compare the predictions to the input results and return the 
	error for each sensor.
	"""

	def __init__(self,cross_validate=False):
		"""
		Create the analysis class
		Setting cross_validate to true will give a better model, but will take longer
		"""
		self.models = {}
		self.cross_validate = cross_validate
		self.defaults = {
			"C": 75,
			"gamma": 0.00105,
			"kernel": "rbf"
		}

		self.cvparams = {
			"kernel": ["rbf"],
			"gamma": linspace(0.0001,0.002,3),
			"C": linspace(50,100,3)
		}
		


	def train(self,data):
		"""
		Train the internal models to predict the behavior of each sensor
		It is important to ensure that the training data is taken from 
		models that are correct.
		"""
		# Read in the data 
		cross_validate = False

		# Choose model parameter search space
		print "Training {0} sensors with {1} rows".format(data.shape[1],data.shape[0])

		for sensor in range(data.shape[1]):
			print "Training model for sensor {0}".format(sensor)
			X,Y = utils.split_xy(data,sensor)

			if self.cross_validate:
				svr = SVR()
				clf = GridSearchCV(svr, self.cvparams, verbose=3)
				model = clf.fit(X, Y)
				print "Best model params for sensor {0}:".format(sensor)
				print model.best_params_
			else:
				C = self.defaults["C"]
				kernel = self.defaults["kernel"]
				gamma = self.defaults["gamma"]
				clf = SVR(C=C, kernel=kernel, gamma=gamma)
				model = clf.fit(X, Y)

			# Check the training RMSE to ensure we are on track
			print "Testing <sensor={0}> model with {1} training rows".format(sensor,data.shape[0])
			Yhat = model.predict(X)
			rmse = utils.rmse(Yhat,Y)
			self.models[sensor] = model
			print "RMSE for <sensor={0}> on training data is {1}".format(sensor,rmse)



	def predict(self,data):
		"""
		Return a column matrix with predictions for all sensors in time
		Assuming the model was perfect the predictions matrix would be identical to @data
		in both shape and values
		"""
		Yhat = np.zeros_like(data)
		npoints = data.shape[0]
		nsensors = data.shape[1]
		for sensor in range(nsensors):
			print "Making predictions for sensor={0} with {1} testing rows".format(sensor,npoints)
			X,Y = utils.split_xy(data,sensor)
			model = self.models[sensor]
			Yhat[:,sensor] = model.predict(X)
		return Yhat



	def error(self,data):
		"""Return the prediction error (RMSE)"""
		Yhat = self.predict(data)
		return Yhat-data



	def plot_predictions(self,time,data):
		"""Plot the predictions against time for each sensor"""
		Yhat = self.predict(data)
		for sensor in range(data.shape[1]):
			y = data[:,sensor]
			yhat = Yhat[:,sensor]
			scale = SCALES[sensor]

			fig = plt.figure()
			plt.plot(time, scale*y, 'b', label=u'Observations')
			plt.plot(time, scale*yhat, 'r', label=u'Predictions')
			plt.xlabel('Time [s]')
			plt.ylabel('Acceleration [g]')
			plt.title('Acceleration Prediction for Sensor {0}'.format(sensor))
			plt.legend(['Recorded values','Predicted Values'])
			plt.xlim([660,700])
			plt.savefig('images/sensor{0}_predictions.png'.format(sensor))



	def plot_error(self,time,data):
		"""Plot the prediction error against time for each sensor"""
		Error = self.error(data)
		for sensor in range(data.shape[1]):
			error = Error[:,sensor]
			scale = SCALES[sensor]

			fig = plt.figure()
			plt.plot(time, scale*error, 'b', label=u'Error')
			plt.xlabel('Time [s]')
			plt.ylabel('Error [g]')
			plt.title('Acceleration Error for Sensor {0}'.format(sensor))
			plt.xlim([660,700])
			plt.savefig('images/sensor{0}_error.png'.format(sensor))



	def plot_predictions_and_error(self,time,data):
		"""Plot the predictions and error against time on subplots"""
		Yhat = self.predict(data)
		for sensor in range(data.shape[1]):
			y = data[:,sensor]
			yhat = Yhat[:,sensor]
			error = yhat - y
			scale = SCALES[sensor]

			fig = plt.figure()
			plt.subplot(2, 1, 1)
			plt.plot(time, scale*y, 'b', label=u'Observations')
			plt.plot(time, scale*yhat, 'r', label=u'Predictions')
			plt.xlabel('Time [s]')
			plt.ylabel('Acceleration [g]')
			plt.title('Acceleration Prediction for Sensor {0}'.format(sensor))
			plt.legend(['Recorded values','Predicted Values'])
			plt.xlim([660,700])
			plt.savefig('images/sensor{0}_predictions.png')

			plt.subplot(2, 1, 2)
			plt.plot(time, scale*error, 'b', label=u'Error')
			plt.xlabel('Time [s]')
			plt.ylabel('Error [g]')
			plt.title('Acceleration Error for Sensor {0}'.format(sensor))
			plt.xlim([660,700])
			plt.savefig('images/sensor{0}_subplot.png'.format(sensor))


	def show_plots(self):
		"""Show the plots drawn so far"""
		plt.show()













