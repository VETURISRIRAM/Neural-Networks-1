import struct
import numpy as np
import numpy.random as random
import matplotlib as mtplot

# Data read from the Label files
label_data= open('train-labels.idx1-ubyte','rb')

#Data read from the 
#a1=img_data.read(4)
#a2=img_data.read(4)
#rowimg=struct.unpack('>I', img_data.read(4))[0]
#colimg=struct.unpack('>I', img_data.read(4))[0]

initialweights = np.zeros((10,784))
for i in range(10):
    x = (1-(-1))*random.sample(784) - 1
    initialweights[i] = x
weight_matrix=np.matrix(np.array(initialweights))
print("Initial Weights = ",weight_matrix)

l1=struct.unpack('>I',label_data.read(4))[0]#Reading first two bytes
l2=struct.unpack('>I',label_data.read(4))[0]

#Generating Desired Outputs
desired=[]
for j in range(60000): #N
    desired.append(struct.unpack('>B',label_data.read(1))[0])

epsilon=0.108
flag=1
error=0
epoch=0
eta=1
n=60000
#matrix_new=[]
while(flag==1):
    error=0
    img_data= open('train-images.idx3-ubyte','rb')
    img_data.read(4)
    img_data.read(4)
    img_data.read(4)
    img_data.read(4)
    for k in range(60000):
        x1=[]
        matrix1=[]
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
        
#    weight_matrix=matrix_new
    epoch+=1
    print("Epoch =  ",epoch," Error = ",error)
    if((error/n)>epsilon):
        flag=1
    else:
        flag=0
print("Epoch=",epoch)
print("Error=",error)

    
        
        