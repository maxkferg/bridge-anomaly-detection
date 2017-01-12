import numpy as np
from numpy import linspace
from numpy import genfromtxt
from scipy.signal import detrend



INDEX = [1,0,4,2,9,6]
MAX_ROWS = 200*1000
SAMPLE_RATE = 200 # Hz
DOUBLE = np.dtype('d')
DATA_FILE = 'data.csv'

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
SCALES = get_scales()



def rmse(predictions, targets):
	"""Return the RMSE for the predictions"""
	return np.sqrt(((predictions - targets) ** 2).mean())


def select_random(matrix,n):
	"""Select random rows from a matrix"""
	nrows = matrix.shape[0]
	index = np.random.randint(nrows,size=n)
	return matrix[index,:]



def sort_by(matrix,column):
	"""Return a matrix sorted by the given column"""
	return matrix[matrix[:,column].argsort()]



def split_xy(data,ycolumn):
	"""
	Split a data matrix into a ycolumn and set of xcolumns
	@ycolumn is the column that will become the yvalues
	return (x,y) tuple, where x is a columm matrix and y is a column vector
	"""
	y = data[:,ycolumn]
	x = np.delete(data,ycolumn,axis=1)
	return (x,y)



def read_data(start,end):
	"""
	Read data from the excel spreadsheet
	Return three time series matrices
	@time. A column vector contaning the time [s]
	@x. A column matrix contaning acceleration from all sensors
	"""
	with open(DATA_FILE,'r') as fd:
		for i in range(start):
			fd.next()
		time,x = read_next(fd, end-start)
	time = time+start/SAMPLE_RATE
	return (time,x)



def read_next(fd,n):
	"""
	Read the next n points from a file
	More performant than read_data
	"""
	print "Reading {0} rows from file".format(n)
	data = genfromtxt(fd, delimiter=',', max_rows=n)
	x = data[:,INDEX]
	x = x/SCALES
	time = np.arange(0,n,dtype=DOUBLE).reshape((n, 1))/SAMPLE_RATE
	return time,x





