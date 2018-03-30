import random
import math
import matplotlib.pylab as pylab
epocharr=[]
msearr=[]
def plot_graph(X,Y):
    pylab.plot(X,Y,'bo')
def plot_graph1(X,Y):
    pylab.title("Curve Fitting")
    pylab.plot(X,Y,'go')
    pylab.show()
def plot_graph2(X,Y):
    pylab.title("Epoch VS MSE")
    pylab.xlabel("Epoch")
    pylab.ylabel("MSE")
    pylab.plot(X,Y,'o-')
    pylab.show()
def generate_I(N):
    X = list()
    for i in range(N):
        X.append(random.uniform(0,1))
    return X
def generate_V(N):
    V = list()
    for j in range(N):
        V.append(random.uniform(-1/10,1/10))
    return V
def generate_DO(N,X,V):
    D = list()
    for k in range(N):
        D.append(math.sin(20*X[k])+(3*X[k])+(V[k]))
    return D
def weights_generate(M,mi,ma):
    W = list()
    for l in range(M):
        W.append(random.uniform(mi,ma))
    return W
def bp(N,M,X,Winputs,Woutputs,Bias,Desired): #Backpropogation_function
    eta = 0.01
    epoch = 0
    bias_output = random.uniform(-1,1)
    flag = True
    while(flag):
        Y = list()
        for i in range(N):
            #Start of Output Calculation 
            Xtemp=X[i]
            val1=list()
            val2=list()
            val3=list()
            for j in range(M):
                 t = (Xtemp * Winputs[j]) + Bias[j]
                 val1.append(t) #U
                 t = (math.tanh(t))
                 val2.append(t) 
                 t = (t*Woutputs[j])
                 val3.append(t) #V
            sum=0
            for k in range(M):
                sum+=val3[k]
            network_out = sum + bias_output
            Y.append(network_out)
            #End of output calculation
            #Start of Backpropogation of network
            for i1 in range(M):
                t1=(eta * (-2 * (Desired[i] - network_out)))
                t2=(1 - math.pow((math.tanh(val1[i1])), 2))
                Winputs[i1]   = Winputs[i1] -  (t1 * t2 * Xtemp * Woutputs[i1])
                Bias[i1]      = Bias[i1] -     (t1 * t2 * Woutputs[i1])
                Woutputs[i1]  = Woutputs[i1] - (t1 * val2[i1])
            bias_output = bias_output - (eta * (-2 * (Desired[i] - network_out)))
        epoch=epoch+1 
        epocharr.append(epoch)
        mse=0
        for q in range(N):
            mse = mse + math.pow((Desired[q] - Y[q]), 2)
        mse = mse/N
        msearr.append(mse)
        print ("Epoch = ",epoch,"  MSE = ",mse)
        if (mse>=0.01):
            flag = True
        else:
            flag = False
    print("Final Value = ",Y)
    print("******BREAK*******")
    print("Weights Inputs = ",Winputs)
    print("Minimum Wii = ",min(Winputs))
    print("Maximum Wii = ",max(Winputs))
    print("******BREAK*******")
    print("Weights Bias = ",Bias)
    print("Minimum Wbb = ",min(Bias))
    print("Maximum Wbb = ",max(Bias))
    print("******BREAK*******")
    print("Weights Outputs = ",Woutputs)
    print("Minimum Woo = ",min(Woutputs))
    print("Maximum Woo = ",max(Woutputs))
    print("******BREAK*******")
    plot_graph(X,Desired)
    plot_graph1(X,Y)
    plot_graph2(epocharr,msearr)
#End of Function       
N=300 #Number of inputs
M=24 #Number of neurons in second layer

I = generate_I(N) #Generate Inputs Xi
V = generate_V(N) #Generate V 
DO = generate_DO(N,I,V) #Calculating Outputs with IO Relationship

#Generating Weights for Input, Output and Bias
W1i = weights_generate(M,-10,10) #-10,10
W1o = weights_generate(M,-7,8) #-7,5
Wb = weights_generate(M,-7,4) #-2,7
bp(N,M,I,W1i,W1o,Wb,DO)