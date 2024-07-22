import pandas as pd 
import numpy as np 
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

def loadData(filename):
    fp = open(filename)
    data = np.array(pd.read_csv(filename,header=None).values)
    lab=["Iris-setosa","Iris-versicolor","Iris-virginica"]
    for i in range(len(data)):
        for j in range(3):
            if data[i][-1]==lab[j]:
                data[i][-1]=j
    data=shuffle(data)
    for i in range(4):
        data[:,i]=(data[:,i]-np.min(data[:,i]))/(np.max(data[:,i])-np.min(data[:,i]))
    train=np.array(data[:-20])
    test=np.array(data[-20:])
    return train,test

def init(inputnum,hiddennum,outnum):
    c=np.random.normal(0.0, 1, (hiddennum, inputnum))
    b=np.random.normal(0.0, 1, (hiddennum, 1))
    w=np.random.normal(0.0, 1, (outnum,hiddennum))
    return c,b,w

def f(x,c,b):
    x=c-x
    X=[]
    for i in range(len(x)):
        X.append(-sum(x[i]*x[i])/(b[i]*b[i]))
    X=np.matrix(X)
    return np.exp(X)

def trainW(c,b,w,x,y,lr):
    hout=f(x,c,b)
    out=np.dot(w,hout)
    err=out-y
    c=np.add(c,-lr*np.dot(np.dot(np.dot((w.T/(b*b)),err),hout.T),(c-x)))
    x_=x-c
    x_=np.matrix([np.sum(i)*np.sum(i) for i in x_]).T
    b=np.add(b,-lr*np.dot(np.dot(np.dot(hout,err.T),(w.T/(b*b*b)).T),x_))
    w-=lr*np.dot(err,hout.T)
    return float(np.linalg.norm(err , ord=1))

def Train():
    train,test=loadData("iris.csv")
    c,b,w=init(4,25,3)
    loss=[]
    acc=[]
    for e in range(300):
        loss_=[]
        for i in train:
            x=i[:-1]
            y=np.zeros((3,1))
            y[i[-1]] += 1
            loss_.append(trainW(c,b,w,x,y,0.01))
        loss.append(sum(loss_)/len(loss_))
        acc_ = []
        for i in test:
            x=i[:-1]
            y=np.zeros((3,1))
            y[i[-1]] += 1
            hout=f(x,c,b)
            out=np.dot(w,hout)
            if(np.argmax(out)==i[-1]):
                acc_.append(1)
            else:
                acc_.append(0)
        acc.append(sum(acc_)/len(acc_))
    plt.subplot(2,1,1)
    plt.plot(loss)
    plt.subplot(2,1,2)
    plt.plot(acc)
    print(acc)
    plt.show()

Train()