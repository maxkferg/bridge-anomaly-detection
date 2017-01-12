import os
import glob
import yaml
import pprint
import numpy as np
from scipy.signal import detrend

CONVERSION_FACTOR = 0.015258789
METADATA_FILE = "TestName.txt"
DATA_PATH = "data"

regex = os.path.join(DATA_PATH,'2014-08*')
folders = glob.glob(regex)

datasets = []



def extract_sensor(line):
    line = line.rstrip()
    number = int(line.split(' ')[2])
    filename = 'U{:02d}_CH0.dat'.format(number)
    return filename

def extract_binary(filename):
    with open(filename,'rb') as content:
        content = content.read().rstrip()
    numbers = content.split('\n')
    numbers = np.array([float(i) for i in numbers]);
    floats = numbers/(2**16) - 2.5*CONVERSION_FACTOR
    return floats


# Isolate the rows which contain acceleration sensors
for folder in folders:
    metadata = os.path.join(folder,METADATA_FILE)
    with open(metadata,'r') as file:
        dataset = {'folder':folder,'sensors':[]}
        started = 0
        while True:
            x = file.readline()
            if 'system 2' in x:
                break
            if started==1:
                dataset['sensors'].append(extract_sensor(x))
            if 'sensing unit(s) used system 1' in x:
                started = 1
        datasets.append(dataset)



# Create a map of sensor:timeseries values
timeseries = []
for dataset in datasets:
    sensors = {}
    folder = dataset['folder']
    for sensor_file in dataset['sensors']:
        path = os.path.join(folder,sensor_file)
        sensors[sensor_file] = extract_binary(path)
    timeseries.append(sensors)


# Create a mapping between sensor and feature index
# At the end of this block we have the following structure:
# {'U51_CH0.dat': 0, 'U162_CH0.dat': 1, 'U76_CH0.dat': 2, 'U234_CH0.dat': 3...}
keys = [t.keys() for t in timeseries]
keys = set([i for sublist in keys for i in sublist])
filemap = dict([(v,k) for (k,v) in enumerate(keys)])
pprint.pprint(filemap)

# Find the number of rows 
rows = sum([t.values()[0].size for t in timeseries])
columns = len(filemap)

# Fill a numpy matrix with time series values
blocks = []
for t in timeseries:
    height = t.values()[0].size
    block = np.zeros(shape=(height,columns))
    for filename,array in t.iteritems():
        column = filemap[filename]
        block[:,column] = array
        block = detrend(block,axis=0)
    blocks.append(block)
data = np.vstack(blocks)

with open('data.csv','wb') as fd:
    #f.write(bytes("SP,"+lists+"\n","UTF-8"))
    np.savetxt(fd,data,delimiter=',')

print data.shape








