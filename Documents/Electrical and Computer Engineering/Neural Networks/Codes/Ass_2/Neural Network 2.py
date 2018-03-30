import struct
import numpy as np
import numpy.random as random
import matplotlib.pyplot as plt
#Weight Function for generating weights
def weightfunc():
    initialweights = np.zeros((10,784))
    for i in range(0,10):
        x = (1-(-1))*random.sample(784) - 1
        initialweights[i] = x
    weight_matrix=np.matrix(np.array(initialweights))
    return weight_matrix
#Function to train images for Multicategory PTA
def TrainingPTA(N1,eta,epsilon,weight_matrix):
    # Data read from the Label files
    label_data= open('train-labels.idx1-ubyte','rb')
    struct.unpack('>I',label_data.read(4))[0]
    struct.unpack('>I',label_data.read(4))[0]
    desired=[]
    for j in range(N1): 
        desired.append(struct.unpack('>B',label_data.read(1))[0])
    flag=1
    error=0
    epoch=0
    epoch_arr=[]
    error_arr=[]
    while(flag==1):
        error=0
        img_data= open('train-images.idx3-ubyte','rb')
        img_data.read(4)
        img_data.read(4)
        img_data.read(4)
        img_data.read(4)
        for k in range(N1):
            x1=[]
            for l in range(784):
                x1.append(struct.unpack('>B',img_data.read(1))[0])
            matrix_inputs=np.matrix(np.array(x1))
            matrix_inputs_transpose=np.transpose(matrix_inputs)
            mm=np.dot(weight_matrix,matrix_inputs_transpose)
            actval=np.argmax(mm)
            newmat=np.zeros((10,1))
            if(desired[k]!=actval):
                newmat[actval]=-1
                newmat[desired[k]]=1
                error=error+1
                weight_matrix=weight_matrix + np.dot((eta*newmat),matrix_inputs)
        epoch+=1
        error_arr.append(error)
        epoch_arr.append(epoch)
        if(epoch==70): break
        print("Epoch =  ",epoch," Error = ",error)
        if((error/N1)>epsilon):
            flag=1
        else:
            flag=0
    print("Epoch=",epoch)
    print("Error=",error)
    plt.title("Graph for Epoch Vs Miscount")
    plt.xlim(0,epoch+1)
    plt.plot(epoch_arr,error_arr,'o-')
    plt.show()
    return(weight_matrix)
#Function to test images for Multicategory PTA
def TestPTA(N2,weight_matrix):
    error_percentage=0
    label_data= open('t10k-labels.idx1-ubyte','rb')
    struct.unpack('>I',label_data.read(4))[0]#Reading first two bytes
    struct.unpack('>I',label_data.read(4))[0]
    desired=[]
    for j in range(N2): 
        desired.append(struct.unpack('>B',label_data.read(1))[0])
    error=0
    img_data= open('t10k-images.idx3-ubyte','rb')
    img_data.read(4)
    img_data.read(4)
    img_data.read(4)
    img_data.read(4)
    for k in range(N2):
        x1=[]
        for l in range(0,784):
            x1.append(struct.unpack('>B',img_data.read(1))[0])
        matrix_inputs=np.matrix(np.array(x1))
        matrix_inputs_transpose=np.transpose(matrix_inputs)
        mm=np.dot(weight_matrix,matrix_inputs_transpose)
        actval=np.argmax(mm)
        newmat=np.zeros((10,1))
        if(desired[k]!=actval):
            newmat[actval]=-1
            newmat[desired[k]]=1
            error=error+1
    error_percentage = (error/N2)*100
    print("Error Percentage = ",error_percentage,"%")

print("Executing all possible combinations")
print("Program Starts")
weight_glob=weightfunc()
print("First Set")
weight_1=TrainingPTA(60000,1,0.106,weight_glob)
TestPTA(10000,weight_1)
print("Second Set")
weight_2=TrainingPTA(50,1,0,weight_glob)
TestPTA(50,weight_2)
print("Third Set")
weight_3=TrainingPTA(50,1,0,weight_glob)
TestPTA(10000,weight_3)
print("Fourth Set")
weight_4=TrainingPTA(1000,1,0,weight_glob)
TestPTA(1000,weight_4)
print("Fifth Set")
weight_5=TrainingPTA(1000,1,0,weight_glob)
TestPTA(10000,weight_5)
print("Part i)i) Set")
weight_glob1=weightfunc()
weight_6=TrainingPTA(60000,1,0.109,weight_glob1)
TestPTA(10000,weight_6)
print("Part i)ii) Set")
weight_glob2=weightfunc()
weight_7=TrainingPTA(60000,1,0.108,weight_glob2)
TestPTA(10000,weight_7)
print("Part i)iii) Set")
weight_glob3=weightfunc()
weight_8=TrainingPTA(60000,1,0.107,weight_glob3)
TestPTA(10000,weight_8)
print("Part h")
weight_9=TrainingPTA(60000,1,0,weight_glob)