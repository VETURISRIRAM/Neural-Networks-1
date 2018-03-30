import numpy as np
import matplotlib.pylab as pylab
eta = 0.01 #Initial Eta Value
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
def a1(x2, y2):
    val1 = ((1 / (1 - x2 - y2)) - 1 / x2)
    return val1
def a2(x3, y3):
    val2 = ((1 / (1 - x3 - y3)) - 1 / y3)
    return val2
while(threshold>0.0001):
    temp1 = init_x-((eta)*(a1(init_x,init_y)))
    temp2 = init_y-((eta)*(a2(init_x,init_y)))
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