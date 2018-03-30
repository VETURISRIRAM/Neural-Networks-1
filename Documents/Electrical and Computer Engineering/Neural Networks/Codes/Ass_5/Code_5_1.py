import struct
import numpy as np
import numpy.random as random
import matplotlib.pylab as pylab

W1=[]#input weight list
W2=[]#output weight list

def weights(inputn,hiddenn,outputn,W1L,W2L,W1U,W2U):
    global W1 
    global W2
    W1 = np.matrix([[random.uniform(W1L, W1U) for i in range(0,hiddenn)] for j in range(0,inputn)])
    W2 = np.matrix([[random.uniform(W2L, W2U) for i in range(0,outputn)] for j in range(0,hiddenn)])
  
def Training(Train_Val,eta,hid):
    global W1
    global W2
    #Opening Label read operation and stores it in desired.
    label_data=open('train-labels.idx1-ubyte','rb')
    struct.unpack('>I',label_data.read(4))[0]
    struct.unpack('>I',label_data.read(4))[0]
    desired=[]
    for k in range(Train_Val): #Will run till required number of Training Samples
        desired.append(struct.unpack('>B',label_data.read(1))[0])
    desiredmat=np.zeros((Train_Val,10))
    for j in range(Train_Val):
        for j2 in range(10):
            if desired[j]==j2:
                desiredmat[j][j2]=1
            else:
                desiredmat[j][j2]=0
     #Training images and reading label operation
    flag=True
    epoch=0
    epoch_arr=[]
    error_arr=[]
    mss_arr=[]
#    for i in range(Train_Val):
    while(flag):
        error=0
        mse=0
        for j in range(Train_Val):
            img_data= open('train-images.idx3-ubyte','rb')
            img_data.read(4)
            img_data.read(4)
            img_data.read(4)
            img_data.read(4)
            temp_read=[]
            for j1 in range(784):
                val=struct.unpack('>B',img_data.read(1))[0]
                val1=(val/256)
                temp_read.append(val1)
            Xi=np.matrix(np.array(temp_read))
            
            V1=np.dot(Xi,W1)
            Out1=np.tanh(V1)
            Out1T=np.transpose(Out1)
            V2=np.transpose(np.dot(Out1,W2))
            Out2=np.tanh(V2)
            ymax=np.argmax(Out2)
            
            desiredmatT=np.transpose(np.matrix(np.array(desiredmat[j])))
            #Delta 2 weight_update
            delta2=np.zeros((10,1))
            t2=(-2)*np.subtract(desiredmatT, Out2)
            for k1 in range(10):
                delta2[k1]=(t2[k1])*(1-np.power(np.tanh(V2[k1]), 2))
            delta2T=np.transpose(delta2)
            
            #delta 1 weight update
            delta1=np.zeros((hid,1))
            temp1=np.dot(W2,delta2)
            V1T=np.transpose(V1)
            for j2 in range(hid): #Number of hidden neurons 
                delta1[j2] = (temp1[j2]) * (1-np.power(np.tanh(V1T[j2]),2))
            
            delta1T=np.transpose(delta1)
            XiT=np.transpose(Xi)
            W2 = W2 - ((eta) * (np.dot(Out1T,delta2T)))
            W1 = W1 - ((eta) * (np.dot(XiT,delta1T)))
            mse = mse + np.power((desired[j] - ymax), 2)
            if (desired[j]!=ymax):
                error=error+1
        print("Epoch = ",epoch,"Error =",error)
        epoch=epoch+1
        limit=error/Train_Val
        msee=mse/Train_Val
        epoch_arr.append(epoch)
        error_arr.append(error)
        mss_arr.append(msee)
        suc=100-(limit*100)
        print("Epoch",epoch,"Error",error,"Success",suc)
    
    pylab.title("Missclassification vs Epoch")
    pylab.xlim(0,epoch+1)
    pylab.plot(epoch_arr,error_arr,'o-')
    pylab.show()
    pylab.title("MSE vs Epoch")
    pylab.xlim(0,epoch+1)
    pylab.plot(epoch_arr,mss_arr,'o-')
    pylab.show()
    print("End of Training")
    if (limit>=0.05):
        flag=True
    else:
        flag=False
           
#Function to test images for Backpropogation Algorithm
def Testing(Train_Val,eta,hid):
    global W1
    global W2
    #Opening Label read operation and stores it in desired.
    label_data=open('t10k-labels.idx1-ubyte','rb')
    struct.unpack('>I',label_data.read(4))[0]
    struct.unpack('>I',label_data.read(4))[0]
    desired=[]
    for k in range(Train_Val): #Will run till required number of Training Samples
        desired.append(struct.unpack('>B',label_data.read(1))[0])
    desiredmat=np.zeros((Train_Val,10))
    for j in range(Train_Val):
        for j2 in range(10):
            if desired[j]==j2:
                desiredmat[j][j2]=1
            else:
                desiredmat[j][j2]=0
    limit=0
    for i in range(1):
        error=0
        for j in range(Train_Val):
            img_data= open('t10k-images.idx3-ubyte','rb')
            img_data.read(4)
            img_data.read(4)
            img_data.read(4)
            img_data.read(4)
            temp_read=[]
            for j1 in range(784):
                val=struct.unpack('>B',img_data.read(1))[0]
                val1=(val/256)
                temp_read.append(val1)
            Xi=np.matrix(np.array(temp_read))
            
            V1=np.dot(Xi,W1)
            Out1=np.tanh(V1)
            V2=np.transpose(np.dot(Out1,W2))
            Out2=np.tanh(V2)
            ymax=np.argmax(Out2)
            
            if (desired[j]!=ymax):
                error=error+1
                
        limit=error/Train_Val
        suc=100-(limit*100)
        print("Error",error,"Success",suc)
        
Hidden=100
W1L=-0.3
W2L=-0.3
W1U=0.3
W2U=0.7
weights(784,Hidden,10,W1L,W2L,W1U,W2U)
Training(60000,0.01,Hidden) #Training Weights for 60000 Images
Testing(10000,0.01,Hidden) #Testing Trained Weights for 10000 Images
