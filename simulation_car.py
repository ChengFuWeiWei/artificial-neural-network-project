import math
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random
import numpy as np
from numpy.core.records import array
import tkinter as tk
from tkinter import filedialog
class MLP:
    def __init__(self,learn_rate,train_round,num_weights):
        self.patience = 20
        self.counter = 0
        self.lastLoss = 1000
        self.learn_rate = learn_rate
        self.train_round = train_round
        self.weights = np.full((1, 4), -1) 
        for i in range(num_weights):   
            hidden_weight = 0.3 * np.random.random_sample((1, 4)) + 0.1
            self.weights = np.append(self.weights,hidden_weight,axis=0)
                   
        self.weights = np.delete(self.weights,0,axis = 0)
        self.output_weight = 0.3 * np.random.random_sample(((num_weights+1),)) + 0.1
    def process_data(self,train_data_path):
        dataset = np.loadtxt(train_data_path)
        D = dataset.shape[1]
        self.training_input = np.full((1, D), -1)
        self.training_output =  np.full((1,), -1)

        for i in dataset:
            inputdata = np.array([i[0:D-1]])
            inputdata = np.insert(inputdata,[0],-1,axis=1)
            outputdata = np.array([i[D-1]])
            
            self.training_input = np.append(self.training_input,inputdata,axis=0)
            self.training_output = np.append(self.training_output,outputdata,axis=0)

        self.training_output = np.delete(self.training_output,0,axis = 0)
        self.training_input = np.delete(self.training_input, 0,axis=0)

        self.maxForward = max(self.training_input[:,1])
        self.minForward = min(self.training_input[:,1])
        self.maxRight = max(self.training_input[:,2])
        self.minRight = min(self.training_input[:,2])
        self.maxLeft = max(self.training_input[:,3])
        self.minLeft = min(self.training_input[:,3])

        for data in self.training_input:
            for i in range(len(data)):
                if i % 4 != 0:
                    if i % 4 == 1:
                        data[i] = self.Normalize(data[i],self.maxForward,self.minForward)
                    elif i % 4 == 2:
                        data[i] = self.Normalize(data[i],self.maxRight,self.minRight)
                    elif i % 4 == 3:
                        data[i] = self.Normalize(data[i],self.maxLeft,self.minLeft)
        return 
    def train(self):
        print('start training')
        Max = max(self.training_output)
        Min = min(self.training_output)
        for turn in range(self.train_round):
            index = 0
            set_O = []
            for train_data in self.training_input:
                #foreward
                y = np.array([-1])
                for weight in self.weights:
                    temp_y = self.sigmoid(np.dot(train_data,weight))
                    y = np.append(y,temp_y)
                    
                O = self.sigmoid(np.dot(y,self.output_weight))
                set_O.append(O)
                #backend
                d = self.Normalize(self.training_output[index], Max,Min)
                delta = (d- O)* O * (1 -O)
                for i in range(1,np.size(y)) :
                    temp_delta = y[i] * (1 - y[i]) * delta * self.output_weight[i]
        
                    self.weights[i -1] = self.weights[i -1] + self.learn_rate * temp_delta * train_data

                self.output_weight = self.output_weight + self.learn_rate * delta * y
                index+=1
            
            if turn % 10 == 0:
                #print('Early Stop')
                newLoss = self.Loss(self.training_output,index,set_O)
                if newLoss < self.lastLoss:
                    #print('reset')
                    self.counter = 0
                    self.lastLoss = newLoss
                else:
                    self.counter+=1
                    if self.counter >= self.patience:
                        print('stop and finish training')
                        break
            
        print('finish training')
        return

    def recognition_rate(self):
        Max = max(self.training_output)
        Min = min(self.training_output)
        count = 0
        for index in range(0,np.size(self.training_output)):
            y = np.array([-1])
            for weight in self.weights:
                temp_y = self.sigmoid(np.dot(self.training_input[index],weight))
                y = np.append(y,temp_y)
            
            O = self.sigmoid(np.dot(y,self.output_weight))
            O = O * (Max - Min) + Min
            if abs(O - self.training_output[index]) <= 5:
                count+=1
        #print(count)
        recognition_rate = count / np.size(self.training_output) * 100
        print(recognition_rate)    
        return
        
    def sigmoid(self,x):
        y = 1 / (1 + np.exp(-x))
        return y
    def Normalize(self,d,Max,Min):
        d = (d - Min) / (Max - Min)
        return d
    def forewawrd(self,data):
        Max = max(self.training_output)
        Min = min(self.training_output)
        y = np.array([-1])
        for weight in self.weights:
            temp_y = self.sigmoid(np.dot(weight,data))
            y = np.append(y,temp_y)
        O = self.sigmoid(np.dot(y,self.output_weight))
        #denormalize
        O = O * (Max - Min) + Min
        return O
    def Loss(self,training_output,index,set_O):
        Max = max(training_output)
        Min = min(training_output)
        #MSE
        sum = 0
        for i in range(index):
            sum = sum + (self.Normalize(training_output[i],Max,Min) - set_O[i])**2

        loss  = sum / index
        return loss 
class cars:
    def __init__(self, position, wheel, horizontal):
        self.position = position  
        self.wheel = wheel
        self.horizontal = horizontal

    def drive(self):
        wheel_radian = math.radians(self.wheel)
        horizontal_radian = math.radians(self.horizontal)
        self.position[0] = self.position[0] + math.cos(wheel_radian+horizontal_radian) + math.sin(wheel_radian)*math.sin(horizontal_radian)
        self.position[1] = self.position[1] + math.sin(wheel_radian+horizontal_radian) - math.sin(wheel_radian)*math.cos(horizontal_radian)
        horizontal_radian = horizontal_radian - math.asin(2*math.sin(wheel_radian)/6)
        self.horizontal = math.degrees(horizontal_radian)

    def turn(self, angle):
        if( (-40) <= angle and angle <= 40 ):
            self.wheel = self.wheel + angle
        else:
            print("illegal wheel angle")

def inrange(point,margin1,margin2):

    x1 = margin1[0]
    x2 = margin2[0]
    if(x1>x2):
        t = x1
        x1 = x2
        x2 = t
    
    y1 = margin1[1]
    y2 = margin2[1]
    if(y1>y2):
        t = y1
        y1 = y2
        y2 = t
    
    if(x1 <= point[0] and point[0] <= x2):
        if(y1 <= point[1] and point[1] <= y2):
            return True
        return False
    else:
        return False

def direction(start,end,horizontal):

    if( -90 < horizontal and horizontal < 0 ):
        if(end[0]-start[0] > 0):
            if(end[1]-start[1] < 0):
                return True
    elif( horizontal == 0 ):
        if(end[0]-start[0] > 0):
                return True
    elif( 0 < horizontal and horizontal < 90 ):
        if(end[0]-start[0] > 0):
            if(end[1]-start[1] > 0):
                return True
    elif( horizontal == 90 ):
         if(end[1]-start[1] > 0):
                return True
    elif( 90 < horizontal and horizontal < 180 ):
        if(end[0]-start[0]<0):
            if(end[1] - start[1] > 0):
                return True
    elif( horizontal == 180 ):
        if(end[0]-start[0] < 0):
                return True
    elif( 180 < horizontal and horizontal < 270 ):
        if(end[0]-start[0]<0):
            if(end[1] - start[1] < 0):
                return True
    elif( horizontal == 270 or horizontal == -90):
        if(end[1]-start[1] < 0):
                return True
    
    return False

def collision(wall,car_position):
    for i in range(len(wall)-1):
        a = 0
        b = 0
        d = 4
        if(wall[i][0]-wall[i+1][0] == 0):
            a = 1
            b = 0
            c = wall[i][0]*-1
        elif(wall[i][1]-wall[i+1][1] == 0):
            a = 0
            b = 1
            c = wall[i][1]*-1
        else:
            a = wall[i][1]-wall[i+1][1]
            b = (wall[i][0]-wall[i+1][0])*-1
            c = (a*wall[i][0]+b*wall[i][1])*-1
        d = abs(a*car_position[0]+b*car_position[1]+c)/(a**2+b**2)**0.5
        print(i,d,[ (b**2*car_position[0]-a*b*car_position[1]-a*c)/(a**2+b**2), (-1*a*b*car_position[0]+a**2*car_position[1]-b*c)/(a**2+b**2) ])
        if(d<3 and inrange([ (b**2*car_position[0]-a*b*car_position[1]-a*c)/(a**2+b**2), (-1*a*b*car_position[0]+a**2*car_position[1]-b*c)/(a**2+b**2) ],wall[i],wall[i+1])):
            return True
    return False

def sensor_distance(wall,car_horizontal,car_position):

    first = True
    min_d=0
    xy=[]
    for i in range(len(wall)-1):
        x = 0
        y= 0
        if(car_horizontal == 90 or car_horizontal == 270 ):
            if(wall[i][0]-wall[i+1][0] == 0):
                continue
            x = car_position[0]
            k2 = (wall[i][1]-wall[i+1][1]) / (wall[i][0]-wall[i+1][0])
            c2 = wall[i][1] - k2 * wall[i][0]
            y = k2 * x + c2
        elif(car_horizontal == 0 or car_horizontal == 180):
            if(wall[i][1]-wall[i+1][1] == 0):
                continue
            y = car_position[1]
            k2 = (wall[i][1]-wall[i+1][1]) / (wall[i][0]-wall[i+1][0])
            c2 = wall[i][1] - k2 * wall[i][0]
            x = (y - c2)/ k2
        elif(wall[i][0]-wall[i+1][0] == 0):
            #car cant be horizontal 
            x = wall[i][0]
            k1 = math.tan(math.radians(car_horizontal))
            c1 = car_position[1] - k1 * car_position[0]
            y = k1 * x + c1
        elif(wall[i][1]-wall[i+1][1] == 0):
            #car cant be horizontal
            y = wall[i][1]
            k1 = math.tan(math.radians(car_horizontal))
            c1 = car_position[1] - k1 * car_position[0]
            x = (y - c1)/ k1     
        else:
            k1 = math.tan(math.radians(car_horizontal))
            k2 = (wall[i][1]-wall[i+1][1]) / (wall[i][0]-wall[i+1][0])
            c1 = car_position[1] - k1 * car_position[0]
            c2 = wall[i][1] - k2 * wall[i][0]
            x = (-1 * (c1 - c2)) / (k1 - k2)
            y = k1 * x + c1

        if(inrange([x,y],wall[i],wall[i+1])):
            if(direction(car_position,[x,y],car_horizontal)):
                #print(i)
                #print(x,y)
                d_x = x -car_position[0]
                d_y = y - car_position[1]
                d = d_x**2 + d_y**2
                d = d ** 0.5
                if(first):
                    min_d = d
                    xy = [x,y]
                    first = False
                    #print(d)
                elif(d<min_d):
                    min_d = d
                    xy = [x,y]
    #print(min_d,xy)
    return(min_d)

def file_path(file_entry):
    file_path = filedialog.askopenfilename()
    file_entry.delete(0,"end")
    file_entry.insert(0, file_path)
def gocar(file_path1,file_path2):
    car = None
    goal = []
    wall = []
    file = open(file_path1,mode='r')
    
    init_parameter = []
    count_line = 1
    for line in file:
        if(count_line==1):
            init_parameter.append(int(line.split(",")[0]))
            init_parameter.append(int(line.split(",")[1]))
            init_parameter.append(0)
            init_parameter.append(int(line.split(",")[2]))
            car = cars( [init_parameter[0],init_parameter[1] ], init_parameter[2]  , init_parameter[3] )
        elif(count_line<=3):
            goal.append( [int(line.split(",")[0]),int(line.split(",")[1])] )
        else:
            wall.append( [int(line.split(",")[0]),int(line.split(",")[1])] )
        count_line = count_line + 1

    plt.ion()

    figure, ax = plt.subplots(figsize=(10, 8))
    sensor_txt = figure.text( 0.02, 0.5, "Front: 0\nRight45: 0\nLeft45: 0", fontsize=10)
    figure.subplots_adjust(left=0.25)
    
    x = car.position[0]
    y = car.position[1]
    point, = ax.plot(car.position[0],car.position[1], color='red', marker='o')
    lbl_point = ax.text(car.position[0],car.position[1],str(car.position),fontsize =10)

    draw_circle = plt.Circle((car.position[0],car.position[1]), 3,fill=False, color='red')
    ax.add_patch(draw_circle)
    
    direct_x = [car.position[0],car.position[0]+ 4*math.cos(math.radians(car.horizontal))]
    direct_y = [car.position[1],car.position[1]+ 4*math.sin(math.radians(car.horizontal))]
    direct, =  ax.plot(direct_x,direct_y, color='red')

    goal_zone = [ goal[0], [goal[0][0],goal[1][1]], goal[1], [goal[1][0],goal[0][1]], goal[0] ]
    
    for i in range(len(goal_zone)-1):
        goalx = [goal_zone[i][0], goal_zone[i+1][0]]
        goaly = [goal_zone[i][1], goal_zone[i+1][1]]
        #print(goalx,goaly)
        ax.plot(goalx,goaly, color='green')

    for i in range(len(wall)-1):
        wallx = [wall[i][0], wall[i+1][0]]
        wally = [wall[i][1], wall[i+1][1]]
        #print(wallx,wally)
        ax.plot(wallx,wally, color='blue')

    learn_rate = 0.1
    train_round= 100
    nums_hidden_weight = 11

    mlp = MLP(learn_rate,train_round,nums_hidden_weight)
    
    mlp.process_data(file_path2)
    mlp.train()
    mlp.recognition_rate()
    while(1):
        
        front_d = mlp.Normalize(sensor_distance(wall,car.horizontal,car.position),mlp.maxForward,mlp.minForward)
        right_d = mlp.Normalize(sensor_distance(wall,car.horizontal-45,car.position),mlp.maxRight,mlp.minRight)
        left_d = mlp.Normalize(sensor_distance(wall,car.horizontal+45,car.position),mlp.maxLeft,mlp.minLeft)

        data = np.array([-1,front_d,right_d,left_d])
        #y = foreward(result.weight,result.weight_out,data)
        car.wheel = mlp.forewawrd(data)

        car.drive()
        if(collision(wall,car.position)):
            break
    
            
        point.set_xdata(car.position[0])
        point.set_ydata(car.position[1])

        lbl_point.set_position((car.position[0], car.position[1]))
        lbl_point.set_text(str(car.position))

        sensor_txt.set_text("Front: " + str(front_d) + "\nRight: " + str(right_d) + "\nLeft: " + str(left_d))

        draw_circle.center = car.position
        
        direct.set_xdata([car.position[0],car.position[0]+ 4*math.cos(math.radians(car.horizontal))])
        direct.set_ydata([car.position[1],car.position[1]+ 4*math.sin(math.radians(car.horizontal))])

        figure.canvas.draw()
        figure.canvas.flush_events()
        if(inrange(car.position,goal[0],goal[1])):
            break
        time.sleep(0.1)

    plt.ioff()
    plt.show()
    
def main():
    window = tk.Tk()
    window.title('simulate car')
    window.geometry('320x160')

    file_label = tk.Label(window,text='軌道資料(.txt)')
    file_label.grid(row = 1, column = 1)
    file_entry = tk.Entry(window)
    file_entry.grid(row = 1, column = 2)
    file_btn = tk.Button(window, text='...',command=lambda: file_path(file_entry))
    file_btn.grid(row=1, column=3)

    train_file_label = tk.Label(window,text='訓練資料(.txt)')
    train_file_label.grid(row = 2, column = 1)
    train_file_entry = tk.Entry(window)
    train_file_entry.grid(row = 2, column = 2)
    train_file_btn = tk.Button(window, text='...',command=lambda: file_path(train_file_entry))
    train_file_btn.grid(row=2, column=3)
    
    check_btn = tk.Button(window, text='啟動',command=lambda: gocar(file_entry.get(),train_file_entry.get()))
    check_btn.grid(row=7, column=2)
    window.mainloop()
if __name__ == '__main__':
    main()