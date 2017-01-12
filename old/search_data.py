import pickle
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import pyplot as plt
from test_data import read_next,get_scales

# Use a simple Gaussian Process Model to predict the acceleration of sensor 1
# Randomly samples a few rows to use as training set
# Test against the same number of randomly selected rows
PREDICT = 2 # Third entry in INDEX
STEP = 100
INDEX = [1,0,4,2,9,6]
SAMPLE_RATE = 200 # Hz
DOUBLE = np.dtype('d')
DATA_FILE = 'data.csv'
MODEL_FILE = 'models/svm-model-{0}'.format(INDEX[PREDICT])

# Load the model from file
with open(MODEL_FILE,'r') as fd:
    print "Opened model file: {0}".format(MODEL_FILE)
    model = pickle.load(fd)

# Make initial predictions
dfd = open(DATA_FILE,'rb')
time,x,y = read_next(dfd, 10*STEP)
yscale = get_scales()[PREDICT]
yhat = model.predict(x)[:,np.newaxis]


fig = plt.figure()
plt.plot(time, yscale*y, 'b', label=u'Observations')
plt.plot(time, yscale*yhat, 'r', label=u'Predictions')
plt.xlabel('Time [s]')
plt.ylabel('Acceleration [g]')
plt.title('Predictions for Training Set')
plt.ylim([-0.1, 0.1])


# Draw 1000 data points starting at point i
def animate(i):
    # Need to read more training data in, and make more predictions
    global time,x,y,yhat
    dt,dx,dy = read_next(dfd, STEP)
    if dx.shape[0]<STEP:
        print "Out of data"
        return

    # Make new predictions    
    dt = dt+max(time)
    dyhat = model.predict(dx)[:,np.newaxis]

    # Stack on the new predictions
    x = np.vstack((x[STEP:,:],dx))
    y = np.vstack((y[STEP:,:],dy))
    time = np.vstack((time[STEP:,:],dt))
    yhat = np.vstack((yhat[STEP:,:],dyhat))

    plt.cla()
    plt.ylim([-0.1, 0.1])
    plt.xlim([min(time), max(time)])
    line1 = plt.plot(time, yscale*y, 'b', label=u'Observations')
    line2 = plt.plot(time, yscale*yhat, 'r', label=u'Predictions')
    plt.xlabel('Time [s]')
    plt.ylabel('Acceleration [g]')
    plt.title('Acceleration Prediction for Testing Set')
    plt.legend(['Recorded values','Predicted Values'])
    print "Current time {0} s".format(time[0])
    return line1 
    

def init():
    return plt.plot(time, yscale*yhat, 'r', label=u'Predictions')

try: 
    duration = int(1000.0/120)*180 # 180 seconds
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=15, metadata=dict(artist='Max'), bitrate=1800)
    ani = animation.FuncAnimation(fig, animate, frames=duration, init_func=init, interval=120, blit=False)
    ani.save('sensor.mp4', writer=writer)
    plt.show()
finally:
    fd.close()



