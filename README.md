# bridge-anomaly-detection
Predict bridge acceleration recordings to detect anomalies



## Usage

The `run.py` file demonstrates how a model can be trained and tested on 
a dataset. This file can be run using:
```
python run.py
```

### utils.read_data(start,end)
The data.csv file contains acceleration data from 12 accelerometers.
This data must be read into a numpy array before it can be used to train or
test the SVR model

```python
from utils import read_data
time,train = read_data(0,70000)
```

### BridgeSensors
The machine learning logic is encapsulated in the BridgeSensors class.
Instances of the bridge sensors class can trained to predict accelerometer readings.

#### BridgeSensors.train(data)
Train the bridgesensors instance to predict acceleration readings

#### BridgeSensors.predict(data)
Use the bridgesensors instance to predict acceleration readings at each accelerometer

#### BridgeSensors.error(data)
Return the difference between acceleration predictions and real data

#### BridgeSensors.plot_error(time,data)
Plot a time series of the errors

#### BridgeSensors.plot_predictions(time,data)
Plot a time series of the predictions

#### BridgeSensors.plot_predictions_and_error(time,data)
Plot the errors and predictions on a subplot

#### BridgeSensors.show_plots(time,data)
Display the plots [blocking]

## License
MIT


