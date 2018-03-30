import random
import numpy as np
import matplotlib.pylab as pylab
X = []
Y = []
X1 = []
Y1 = []
sumx=0
sumy=0
diff1=[]
diff2=[]
xy = 0
xx = 0
temp=0
for i in range(50):
    X.append(i)
    sumx+=i
avgx=sumx/50
print(X)
print(avgx)
for j in range(50):
    temp=random.uniform(-1,1)
    t1=j+temp
    Y.append(t1)
    sumy+=t1
avgy=sumy/50
print(Y)
print(avgy)
for i1 in range(50):
    temp = X[i1] - avgx
    diff1.append(temp)
for j1 in range(50):
    temp = Y[j1] - avgy
    diff2.append(temp)
xy = 0
xx = 0
for i2 in range(50):
    p = diff1[i2] * diff2[i2]
    xy = xy + p
for j2 in range(50):
    q = diff1[j2] * diff2[j2]
    xx = xx + q
w1 = xy / xx
w0 = avgy - (w1*avgx)
print("Weights to minimize function:")
print(w0, w1)
X1 = np.array(range(0, 50))
Y1 = (w0 + (w1 * X1))
pylab.scatter(X, Y, color='r')
pylab.plot(X1, Y1,'g')
pylab.show()