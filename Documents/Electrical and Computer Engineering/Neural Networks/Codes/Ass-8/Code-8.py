import numpy
import random
import math
import collections
import matplotlib.pyplot as plt
from cvxopt import matrix,solvers

N=100
X=[]
desired=[]
class0=[]
class1=[]
alpha=[]
svx=[]
svy=[]
Hma=[]
Hmi=[]
H=[]
xc1 = []
yc1 = []
xc0 = []
yc0 = []

#Function for getting X values
def input_plot(N):
    x=[]
    for i in range(N):
        x1=[]
        for j in range(2):
            temp=random.uniform(0,1)
            x1.append(temp)
        x.append(x1)
    x=numpy.array(x)
    return x

#Function to plot the scatter points
def plot(N,X):
    
    plt.title("Plot")
    for i in range(N):
        plt.plot(X[i][0],X[i][1],'o')
    plt.show()

#Classifying the inputs as Sun and Mountains    
def desired_outputs(N,X):
    plt.title("Fig1. Inputs")  
    for i in range(N):
        if((X[i][1]) < (((1/2)*(numpy.sin((10)*(X[i][0])))) + 0.3 )  or ((math.pow((X[i][1]-0.5),2) + math.pow((X[i][0]-0.5),2)) < math.pow(0.15,2))):
            desired.append(1)
            class1.append(X[i])
            plt.plot(X[i][0],X[i][1],'go')
        else:
            desired.append(-1)
            class0.append(X[i])
            plt.plot(X[i][0],X[i][1],'rx')
    plt.show()

#Polynomial Kernel Function
def kernel(x,y,p=5):
	return(1 + numpy.dot(x,y)) ** p

#Function to calculate alpha value using CVXOPT.solver.qp
def cal_alpha(N,X):
	polmat=[]
	for i in range(len(X)):
		polmat.append([])
		for j in range(len(X)):
			polmat[i].append(kernel(X[i],X[j]))

	P=[]
	for i in range(0,N):
		P.append=[]
		for j in range(0,N):
			if i!=j:
				P[i].append(((polmat[i][j]  +  polmat [j][i]) * desired[i] * desired[j]) / 2 )
			else:
				P[i].append(polmat[i][j] * desired[i] * desired[j])
	p=numpy.array(P)

	G=[]
	for i in range(0,N):
		G.append([])
		for j in range(0,N):
			if i!=j:
				G[i].append(0.0)
			else:
				G[i].append(-1)
	G=numpy.array(G)

	A=[]
	for i in range(0,N):
		A.append(desired[i] * 1.0)
	A=numpy.array(A)
	A.resize(1,N)

	q=matrix(numpy.ones(100)* -1)

	h=numpy.zeros(100)
	h.resize(100,1)

	P=(0.5)*matrix(P)
	q=matrix(q)
	G=matrix(G)
	h=matrix(h)
	A=matrix(A)
	b=matrix(0.0)

	sol=solvers.qp(P,q,G,h,A,b)#maximising Alpha
	alphamat=sol['x']
	alphalist=numpy.ravel(alphamat)

	return alphalist

#Getting Support Vector Machines
def support_v(alphalist):
	print("Support Vectors")
	for i in range(0,len(alphalist)):
		if(alphalist[i] > 90):
			temp=X[i]
			print (temp)
			svx.append(temp[0])
			svy.append(temp[1])
	print("Number of Support Vectors:",len(svx))

#Calculating Theta Value
def calculate_theta(X):
	sv1=X[32]
	svxx=sv[0]
	svyy=sv[1]
	theta=0.0
	sum=0.0

	for i in range(0,len(X)):
		a=alpha[i] * desired[i] * kernel(X[i],sv1)
		sum=sum+a
	print("Sum = ",sum)
	theta=desired[32]-sum
	print(theta)
	return theta

def mul():
	ranx=numpy.linspace(0,1,1000)
	rany=numpy.linspace(0,1,1000)
	ran=[]
	for i in range(1000):
		for j in range(1000):
			ran.append(ranx[i],rany[j])
	return ran
#G(X)
def g(r,X):
	sum=0.0
	for i in range(0,len(X)):
		temp=alpha[i] * desired[i] * kernel(X[i],r)
		sum=sum+temp
	theta=calculate_theta(X)
	return(sum+theta)

#Execution Block of Program
X=input_plot(N)
X=numpy.matrix(X)

plot(N,X)
desired_outputs(N,X)

alpha=cal_alpha(N,X)
print("Alpha = ",alpha)

support_v(alpha,X)
ran1=mul()

for i in range(0,len(ran1)):#Hyperplane Classification
	val=g(ran1,X)
	if val < 0.1 and val > -0.1:
		H.append(ran1[i])
	if val < 1.1 and val > 0.9:
		Hma.append(ran1[i])
	if val < -0.9 and val > -1.1:
		Hmi.append(ran1[i])

#Plotting the Hyperplanes
for el in class1:
    xc1.append(el[0])
    yc1.append(el[1])
    
for el in class0:
    xc0.append(el[0])
    yc0.append(el[1])

fig, ax = plt.subplots(figsize=(10,10))
plt.title("Final Plot")
plt.scatter(xc0,yc0, c = 'red')
plt.scatter(xc1,yc1, c = 'green')
plt.scatter(*zip(*Hma), c = 'red',s = 1)
plt.scatter(*zip(*H), c = 'black', s = 1)
plt.scatter(*zip(*Hmi), c = 'red',s = 1)
plt.show()
