import struct
import random
import numpy as NP
import matplotlib.pylab as mpl

globalW = list()
globalMisCalc = list()

def ReadData(startN, N, bTrain):

    if bTrain:
        fImages = open('train-images.idx3-ubyte', 'rb')
        fLabels = open('train-labels.idx1-ubyte', 'rb')
    else:
        fImages = open('t10k-images.idx3-ubyte', 'rb')
        fLabels = open('t10k-labels.idx1-ubyte', 'rb')

    s1, s2, s3, s4 = fImages.read(4), fImages.read(4), fImages.read(4), fImages.read(4)
    mnIm = struct.unpack('>I', s1)[0]
    numIm = struct.unpack('>I', s2)[0]
    rowsIm = struct.unpack('>I', s3)[0]
    colsIm = struct.unpack('>I', s4)[0]
    fImages.seek(16 + startN * rowsIm * colsIm)

    mnL = struct.unpack('>I', fLabels.read(4))[0]
    numL = struct.unpack('>I', fLabels.read(4))[0]
    fLabels.seek(8 + startN)

    T = []

    for b in range(0, N):

        x = []
        for i in range(0, rowsIm * colsIm):
            val = struct.unpack('>B', fImages.read(1))[0]
            x.append(val / 255.0)

        val = struct.unpack('>B', fLabels.read(1))[0]
        y = []
        for i in range(0, 10):
            if val == i:
                y.append(1)
            else:
                y.append(0)


        T.append((x, y))

    fImages.close()
    fLabels.close()
    return T

def multiCategoryPerceptron(T, N):
    W = generateWeights()
    epoch = 0
    global globalMisCalc
    errors = list()
    epsilon = 0.5
    eta = 10
    V = []
    dX = []
    flag = True
    while flag:
        count = 0
        for i in T:
            Xi = NP.matrix(i[0])
            Vi = list()
            for w in W:
                Wi = NP.matrix(w)
                Wi = NP.transpose(Wi)
                ans = (NP.dot(Xi, Wi))
                Vi.append(ans.item(0))
            j = NP.argmax(Vi)
            dXi = generatedXi(j)
            dX.append(dXi)
            V.append(Vi)
            if not NP.array_equal(NP.array(dXi), NP.array(i[1])):
                count = count + 1
        errors.append(count)
        globalMisCalc.append([epoch, count])
        print("Epoch = {}".format(epoch))
        print("Errors = {}".format(errors))
        epoch += 1
        x = int(errors[epoch - 1]) / N
        print(x * 100)
        Wup = updateWeights(W, eta, V, T)
        W = Wup
        global globalW
        globalW = W
        if x > epsilon:
            flag = True
        else:
            flag = False

def updateWeights(W, eta, V, T):

    Wxi = list()
    result = list()
    Wup = W
    for vi in V:
        temp = list()
        for i in range(0, len(vi)):
            if vi[i] >= 0:
                temp.append(1)
            else:
                temp.append(0)
        Wxi.append([temp])
    for i in range(0, len(T)):
        X = T[i][1]
        temp1 = NP.subtract(X, Wxi[i])
        temp1 = NP.dot(eta, temp1)
        result.append(temp1)
    for i in range(0, len(T)):
        X = T[i][0]
        R = result[i][0]
        Wupdate = list()
        for r in R:
            temp2 = list()
            for k in X:
                temp2.append(r * k)
            Wupdate.append(temp2)
        Wup = Wup + Wupdate
    return Wup

def generatedXi(j):
    dXi = list()
    for i in range(0, 10):
        if i == j:
            dXi.append(1)
        else:
            dXi.append(0)
    return dXi

def generateWeights():
    W = list()
    temp = list()
    for i in range(10):
        temp = [random.random() for i in range(784)]
        W.append(temp)
    return W

def multiCategoryPerceptronTestData(T):
    print(len(T))
    W = globalW
    epoch = 0
    errors = list()
    V = []
    dX = []

    count = 0
    for i in T:
        Xi = NP.matrix(i[0])
        Vi = list()
        for w in W:
            Wi = NP.matrix(w)
            Wi = NP.transpose(Wi)
            ans = (NP.dot(Xi, Wi))
            Vi.append(ans.item(0))
        j = NP.argmax(Vi)
        dXi = generatedXi(j)
        dX.append(dXi)
        V.append(Vi)
        if not NP.array_equal(NP.array(dXi), NP.array(i[1])):
            count = count + 1
    errors.append(count)
    print("Epoch = {}".format(epoch))
    print("Errors = {}".format(errors))
    x = int(errors[epoch - 1]) / N
    print("Percentage of Miscalculations")
    print(x * 100)

def plot_epoch_misCal():
    xAxis = list()
    yAxis = list()

    for i in globalMisCalc:
        xAxis.append(i[0])
        yAxis.append(i[1])
    mpl.scatter(xAxis, yAxis, color='g')
    mpl.xlabel("Epoch")
    mpl.ylabel("Miscalculations")
    mpl.title("Epoch Vs Miscalculations")
    x1 = NP.array(range(-1, 50))
    y1 = NP.array(range(-1, 50))
    mpl.grid(True)
    mpl.show()

print("Program has started")
N = 60000
multiCategoryPerceptron(ReadData(0, N, True), N)
plot_epoch_misCal()

N = 10000
multiCategoryPerceptronTestData(ReadData(0, N, False))