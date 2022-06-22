import numpy as np
from numpy.core.fromnumeric import shape
import pandas as pd



data_path = 'C:\\Users\\Wei\\Desktop\\類神經網路_HW2\\train4dAll.txt'
dataset = np.loadtxt(data_path)
D = dataset.shape[1]
training_input = np.full((1, D), -1)
training_output =  np.full((1,), -1)

for i in dataset:
    inputdata = np.array([i[0:D-1]])
    inputdata = np.insert(inputdata,[0],-1,axis=1)
    outputdata = np.array([i[D-1]])
    
    training_input = np.append(training_input,inputdata,axis=0)
    training_output = np.append(training_output,outputdata,axis=0)

training_output = np.delete(training_output,0,axis = 0)
training_input = np.delete(training_input, 0,axis=0)
