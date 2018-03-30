import struct
import numpy.random as random
import numpy as np
import matplotlib.pylab as pylab
import math

hidden=50
w1N=[]
w2N=[]
eta = 0.01

def weights(HN,N,L,U):
    W = np.zeros((HN,N))
    for i in range(HN):
        temp = (U-L)*random.sample(N) + L         
        W[i] = temp
    W_mat=np.matrix(np.array(W))
    return W_mat

def train_backprop(dataset):
    global w1N
    global w2N
    train_lbl = open('train-labels.idx1-ubyte', 'rb')
    for i in range(2):
        train_lbl.read(4)    
    
    d_list=[]
    des = np.zeros((dataset,10))
    for i in range (dataset):
        index=struct.unpack('>B', train_lbl.read(1))[0]
        d_list.append(index)
        des[i][index]=1
    epoch=0
    thres=1
    epoch_list=[]
    error_list=[]
    mss_list=[]
    while(thres>0.02):
        train_img = open('train-images.idx3-ubyte', 'rb')
        for i in range(4):
            train_img.read(4)
        error = 0
        mse = 0
        for i in range(dataset): 
            x=[]
            for p in range(784):
                x.append((struct.unpack('>B', train_img.read(1))[0])/255.0)
            x=np.matrix(np.array(x))
            x_T=np.transpose(x)
        
            v1 = np.dot(w1N,x_T)
            z=np.tanh(v1)
            z_T=np.transpose(z)
            v2 = np.transpose(np.dot(z_T,w2N))
            y=np.tanh(v2)
            
            ymax=np.argmax(y)
            delta2= np.zeros((10,1))
            des_T=np.transpose(np.matrix(np.array(des[i])))
    
            sub= -2*np.subtract(des_T,y)
    
            for h in range(10):
                delta2[h]= sub[h]*(1-math.pow(np.tanh(v2[h]),2))      
            delta2_T=np.transpose(delta2)
            
            delta1= np.zeros((hidden,1)) # delta1 = 50X1
            mul = np.dot(w2N,delta2) # w2N = 50X10 , delta2 = 10X1 , mul = 50X1
            for j in range(hidden):
                delta1[j]=mul[j]*(1-math.pow(np.tanh(v1[j]),2))
             
            w2N = w2N - eta*np.dot(z,delta2_T)
            w1N = w1N - eta*np.dot(delta1,x)
            mse = mse + math.pow((d_list[i] - ymax), 2)
            if d_list[i]!=ymax:
                error = error +1
        thres=error/dataset
        msee=mse/dataset
        epoch=epoch+1
        epoch_list.append(epoch)
        error_list.append(error)
        mss_list.append(msee)
        suc=100-(thres*100)
        print("Epoch",epoch,"Error",error,"Success",suc)
    
#    print("w1N minimum",w1N.min())
#    print("w1N maximum",w1N.max())
#    print("w2N minimum",w2N.min())
#    print("w2N maximum",w2N.max())
    pylab.title("Missclassification vs Epoch")
    pylab.xlim(0,epoch+1)
    pylab.plot(epoch_list,error_list,'o-')
    pylab.show()
    pylab.title("MSE vs Epoch")
    pylab.xlim(0,epoch+1)
    pylab.plot(epoch_list,mss_list,'o-')
    pylab.show()
    
def test_backprop(dataset):
    global w1N
    global w2N
    train_lbl = open('t10k-labels.idx1-ubyte', 'rb')
    for i in range(2):
        train_lbl.read(4)    
    
    d_list=[]
    des = np.zeros((dataset,10))
    for i in range (dataset):
        index=struct.unpack('>B', train_lbl.read(1))[0]
        d_list.append(index)
        des[i][index]=1
    thres=0
    for j in range(1):
        train_img = open('t10k-images.idx3-ubyte', 'rb')
        for i in range(4):
            train_img.read(4)
        error = 0
        for i in range(dataset): 
            x=[]
            for p in range(784):
                x.append((struct.unpack('>B', train_img.read(1))[0])/255.0)
            x=np.matrix(np.array(x))
            x_T=np.transpose(x)
            v1 = np.dot(w1N,x_T)
            z=np.tanh(v1)
            z_T=np.transpose(z)
            v2 = np.transpose(np.dot(z_T,w2N))
            y=np.tanh(v2)
            ymax=np.argmax(y)
            
            if d_list[i]!=ymax:
                error = error +1
        thres=error/dataset
        suc=100-(thres*100)
    print("Error",error,"Success",suc)

w1N = weights(hidden,784,-0.3,0.3) # H X ip
w2N = weights(hidden,10,-0.3,0.7)   # H X op
train_backprop(60000)       
test_backprop(10000)