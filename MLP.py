from typing import Counter
import numpy as np
from numpy.core.numeric import count_nonzero
from numpy.lib.function_base import delete


class MLP:
    def __init__(self,learn_rate,train_round,num_weights):
        self.patience = 20
        self.counter = 0
        self.lastLoss = 1000
        self.learn_rate = learn_rate
        self.train_round = train_round
        self.weights = np.full((1, 4), -1) 
        for i in range(num_weights):   
            hidden_weight = 0.2 * np.random.random_sample((1, 4)) + 0
            self.weights = np.append(self.weights,hidden_weight,axis=0)
                   
        self.weights = np.delete(self.weights,0,axis = 0)
        self.output_weight = 0.2 * np.random.random_sample(((num_weights+1),)) + 0
        
    def train(self,training_input,training_output):
        Max = max(training_output)
        Min = min(training_output)
        for turn in range(self.train_round):
            index = 0
            set_O = []
            for train_data in training_input:
                #forward
                y = np.array([-1])
                for weight in self.weights:
                    
                    temp_y = self.sigmoid(np.dot(train_data,weight))
                    y = np.append(y,temp_y)
                    
                O = self.sigmoid(np.dot(y,self.output_weight))
                #print('O:',O)
                set_O.append(O)
                #backend
                d = self.Normalize(training_output[index], Max,Min)
                delta = (d- O)* O * (1 -O)
                '''
                print('w1:',self.weights[0])
                print('w2:',self.weights[1])
                print('w3:',self.output_weight)
                print('delta:',delta)
                print('y:',y)
                '''
                for i in range(1,np.size(y)) :
                    temp_delta = y[i] * (1 - y[i]) * delta * self.output_weight[i]
                    self.weights[i -1] = self.weights[i -1] + self.learn_rate * temp_delta * train_data
                    #print(temp_delta)
                    #print(self.weights[i -1])

                self.output_weight = self.output_weight + self.learn_rate * delta * y
                index+=1
                
           
            if turn % 10 == 0:
                #print('Early Stop')
                newLoss = self.Loss(training_output,index,set_O)
                if newLoss < self.lastLoss:
                    #print('reset')
                    self.counter = 0
                    turn_best = turn
                    self.lastLoss = newLoss
                else:
                    self.counter+=1
                    print(self.counter)
                    if self.counter >= self.patience:
                        print('stop training')
                        return self.output_weight

        return self.output_weight
    def Loss(self,training_output,index,set_O):
        Max = max(training_output)
        Min = min(training_output)
        #MSE
        sum = 0
        for i in range(index):
            sum = sum + (self.Normalize(training_output[i],Max,Min) - set_O[i])**2

        loss  = sum / index
        return loss
    def recognition_rate(self,training_input,training_output):
        Max = max(training_output)
        Min = min(training_output)
        count = 0
        for index in range(0,np.size(training_output)):
            y = np.array([-1])
            for weight in self.weights:
                temp_y = self.sigmoid(np.dot(training_input[index],weight))
                y = np.append(y,temp_y)
            
            O = self.sigmoid(np.dot(y,self.output_weight))
            O = O * (Max - Min) + Min
            if abs(O - training_output[index]) <= 5:
                count+=1
        
        recognition_rate = count / np.size(training_output) * 100
        print(recognition_rate)
            
        return
        
    def sigmoid(self,x):
        y = 1 / (1 + np.exp(-x))
        return y
    def Normalize(self,d,Max,Min):
        d = (d - Min) / (Max - Min)
        return d


#learn_rate = ?
#train_round = 1000
#nums_hidden_weight = 11
mlp = MLP(0.2,100,11)
train_data_path = 'C:\\Users\\Wei\\Desktop\\類神經網路_HW2\\train.txt'
train_dataset = np.loadtxt(train_data_path)

validate_data_path = 'C:\\Users\\Wei\\Desktop\\類神經網路_HW2\\validate.txt'
validate_dataset = np.loadtxt(validate_data_path)

D = train_dataset.shape[1]
training_input = np.full((1, D), -1)
training_output =  np.full((1,), -1)

validate_input = np.full((1, D), -1)
validate_output =  np.full((1,), -1)

for i in train_dataset:
    inputdata = np.array([i[0:D-1]])
    inputdata = np.insert(inputdata,[0],-1,axis=1)
    outputdata = np.array([i[D-1]])
        
    training_input = np.append(training_input,inputdata,axis=0)
    training_output = np.append(training_output,outputdata,axis=0)

training_output = np.delete(training_output,0,axis = 0)
training_input = np.delete(training_input, 0,axis=0)

maxForward = max(training_input[:,1])
minForward = min(training_input[:,1])
maxRight = max(training_input[:,2])
minRight = min(training_input[:,2])
maxLeft = max(training_input[:,3])
minLeft = min(training_input[:,3])

for data in training_input:
    for i in range(len(data)):
        if i % 4 != 0:
            if i % 4 == 1:
                data[i] = mlp.Normalize(data[i],maxForward,minForward)
            elif i % 4 == 2:
                data[i] = mlp.Normalize(data[i],maxRight,minRight)
            elif i % 4 == 3:
                data[i] = mlp.Normalize(data[i],maxLeft,minLeft)
    
'''
for data in training_input:
    for num in data:
        num = mlp.Normalize()
'''
'''
for i in validate_dataset:
    inputdata = np.array([i[0:D-1]])
    inputdata = np.insert(inputdata,[0],-1,axis=1)
    outputdata = np.array([i[D-1]])
        
    validate_input = np.append(validate_input,inputdata,axis=0)
    validate_output = np.append(validate_output,outputdata,axis=0)

validate_input = np.delete(validate_input,0,axis = 0)
validate_output = np.delete(validate_output, 0,axis=0)
'''


weight = mlp.train(training_input,training_output)


mlp.recognition_rate(training_input,training_output)
