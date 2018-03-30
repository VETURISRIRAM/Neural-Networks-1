import numpy as np
import matplotlib.pylab as pylab
from numpy.linalg import inv
eta = .01 #Initial Eta Value
init_x=0.1
init_y=0.35
x_values=[]
y_values=[]
energy_matrix=[]
i=0
threshold=1
def func(x, y):#Energy Function
    val=(-(np.log10(1-x-y)))-np.log10(x)-np.log10(y)
    return val
def a1(x2, y2):#Function to generate 1st element of g matrix
    val1 = ((1 / (1 - x2 - y2)) - 1 / x2)
    return val1
def a2(x3, y3):#Function to generate 2nd element of g matrix
    val2 = ((1 / (1 - x3 - y3)) - 1 / y3)
    return val2
def hmx(x, y):#Function to form Hessian double derivative matrix and take inverse of it
    t1=1-x-y
    t2 = np.matrix([[((1/t1)**2) + ((1/x)**2), ((1/t1)**2)],
                       [((1/t1)**2), ((1/(t1))**2)+((1/y)**2)]])
    inver = inv(np.matrix(t2))
    return inver
while(threshold>0.0001):
    g = [[a1(init_x,init_y)], [a2(init_x, init_y)]]
    hessianmat = np.dot(hmx(init_x,init_y), g)
    temp1 = init_x - (eta*hessianmat.item(0))
    temp2 = init_y - (eta*hessianmat.item(1))
    energy_matrix.append(func(temp1, temp2))
    x_values.append(temp1)
    y_values.append(temp2)
    threshold= abs(temp1-init_x)
    init_x = temp1
    init_y = temp2
    i+=1
pylab.title("Gradient Descent Function")
pylab.plot(x_values,y_values,'bo-')
pylab.show()
pylab.title("Energy Function Graph")
pylab.plot(energy_matrix,'go-')
pylab.show()
print("Number of iterations",i)