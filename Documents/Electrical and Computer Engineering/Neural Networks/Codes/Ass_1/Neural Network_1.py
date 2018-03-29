import random
import matplotlib.pyplot as plt
#Generating Weights
w0=random.uniform(-1/4,1/4) 
w1=random.uniform(-1,1)
w2=random.uniform(-1,1)
w10=random.uniform(-1,1) 
w11=random.uniform(-1,1)
w12=random.uniform(-1,1)
#Function to do task for differnet eta and sample size
def perceptron(sample_size,eta):
    arr=[]
    for i in range(sample_size):
        arr.append([])
        arr[i].append(1)
        arr[i].append(random.uniform(-1,1))
        arr[i].append(random.uniform(-1,1))
    #Putting values of initial weights into weights array
    weights0=[]
    weights0.append(w0)
    weights0.append(w1)
    weights0.append(w2)
    print (weights0)
    print("W0= ",w0)
    print("W1= ",w1)
    print("W2= ",w2)
    print (arr)
    print ("___________________BREAK_________________")
    #Classifying variables into S0,S1 Regions from S
    s0 = []
    s1 = []
    count1=0
    count2=0
    do=[]
    for j in arr:
        x1 = j[0]
        x2 = j[1]
        x3 = j[2]
        mulmat = ((x1*weights0[0])+(x2*weights0[1])+(x3*weights0[2]))
        if mulmat >= 0:
            del j[0]
            do.append(1) #Generating desired outputs
            count1=count1+1
            s0.append(j) 
        elif mulmat < 0:
            del j[0]
            count2=count2+1
            s1.append(j)
            do.append(0) #Generating desired outputs
    print ("Desired Output", do)
    print (s0)
    print (count1)
    print (s1)
    print (count2)
    #print the plot
    for xp1 in s0:
        tmp1=xp1[0]
        tmp2=xp1[1]
        plt.plot(tmp1,tmp2,'ro')
    for xp2 in s1:
        tp1=xp2[0]
        tp2=xp2[1]
        plt.plot(tp1,tp2,'bo')
    #Setting the upper and lower limits for the graph
    for l in arr:
        y1 = ((-w0-w1)/w2)
        y2 = ((-w0+w1)/w2)
        xvminmax = [1,-1]
        yvminmax = [y1,y2]
        plt.plot(xvminmax, yvminmax, 'b-',lw=2)
    plt.axis([-1, 1, -1, 1])
    plt.ylim(-1,1)
    plt.xlim(-1,1)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()    
    print("--------------------BREAK-----------------------")
    #Perceptron Training Algorithm
    print("W10= ",w10)
    print("W11= ",w11)
    print("W12= ",w12)
    #Weights for the new 
    weights1=[]
    weights1.append(w10)
    weights1.append(w11)
    weights1.append(w12)
    print (weights1)
    flag=1 #Flag value for while loop
    missarr=[] #Counting Array of Miss Count
    epocharr=[] #Epoch array to store epoch values
    epoch=0
    weightsu=weights1[0]
    weightsu1=weights1[1]
    weightsu2=weights1[2]
    wlist=[]
    cmiss=0
    mat=0
    output=0
    #Multiplcation of the weights with the inputs
    while flag == 1:
        epoch = epoch + 1
        epocharr.append(epoch)
        i1=0
        for q in arr:
            t11=1
            q1 = [t11] + q
            x11 = q1[0]
            x12 = q1[1]
            x13 = q1[2]
            mat = ((x11*weightsu) + (x12*weightsu1) + (x13*weightsu2))
            if  mat >= 0 :
                output=1
            elif mat < 0 :
                output=0
            if (do[i] != output):
                print("Update")
                weightsu=((weightsu) + (eta * (x11) * (do[i1] - output)))
                weightsu1=((weightsu1) + (eta * (x12) * (do[i1] - output)))
                weightsu2=((weightsu2) + (eta * (x13) * (do[i1] - output)))
                cmiss=cmiss+1
                i1=i1+1
        epoch = epoch + 1
        epocharr.append(epoch)        
        missarr.append(cmiss)
        wlist.append([])
        wlist[epoch].append(weightsu)
        wlist[epoch].append(weightsu1)
        wlist[epoch].append(weightsu2)
        print("Epoch No for convergence: ",epoch)
        if(cmiss == 0):
           flag=0
        else :
            flag=1  
    print("Weights after 1st epoch= ",wlist[0])
    print("Final weights: ",wlist[epoch-1])
    print("Miscount List for each Epoch:",missarr)
    print("Epoch Values:",epocharr)
    plt.title("Graph for Epoch Vs Miscount") #Graph for Epoch Vs Number of Misclassifications
    plt.axis([0,epoch+1,0, 100])
    plt.plot(epocharr,missarr,'o-')
    plt.show()
    return    
print("100 Samples PTA Eta -1")    
perceptron(100,1)
print("100 Samples PTA Eta -.1")   
perceptron(100,.1)
print("100 Samples PTA Eta -10")    
perceptron(100,10)

print("1000 Samples PTA Eta - 1")    
perceptron(1000,1)
print("1000 Samples PTA Eta - .1")   
perceptron(1000,.1)
print("1000 Samples PTA Eta - 10")    
perceptron(1000,10)

