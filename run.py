from utils import read_data
from analytics import BridgeSensors

# Read the training data from the csv file
time,train = read_data(0,70000)

# Read the testing data from the csv file
time,test = read_data(130000,162000)

# Train a sensor prediction model
sensors = BridgeSensors(cross_validate=False)
sensors.train(train)

# Obtain the predictions and errors
# These could be used for plotting or anomaly detection
print "Calculating predictions for all sensors"
predictions = sensors.predict(test)
errors = sensors.error(test)

# Plot the prections and prediction error on test data
#print "Plotting the predictions for all sensors"
#sensors.plot_predictions(time,test)
#sensors.show_plots()

# Plot the error on test data
#print "Plotting the prediction error for all sensors"
#sensors.plot_error(time,test)
#sensors.show_plots()

# Plot the predictions/error on test data
sensors.plot_predictions_and_error(time,test)
sensors.show_plots()