import pandas as pd 
import numpy as np 
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
def loadData(iris):
    data = np.array(pd.read_csv(iris,header=None).values)
    lab=["Iris-setosa","Iris-versicolor","Iris-virginica"]
    for i in range(len(data)):
        for j in range(len(lab)):
            if data[i][-1]==lab[j]:
                data[i][-1]=j
    data=shuffle(data)
    for i in range(4):
        data[:,i]=(data[:,i]-np.min(data[:,i]))/(np.max(data[:,i])-np.min(data[:,i]))
    train=np.array(data[:-20])
    test=np.array(data[-20:])
    return train,test

#建立一个4个输入节点，3个输出节点，10个隐藏节点的RBFNN
def init(inputnum,hiddennum,outnum):
    c=np.random.normal(0.0, 1, (hiddennum, inputnum))
    b=np.random.normal(0.0, 1, (hiddennum, 1))
    return c,b

def f(x,c,b):
    x=c-x #样本点1*4与基函数中心10*4相减以计算距离，相减之后的x是10*4
    X=[]
    for i in range(len(x)):
        X.append(-sum(x[i]*x[i])/(b[i]*b[i])) #方差10*1
    X=np.asmatrix(X) #10*1
    return np.exp(X)

def Train():
    c,b=init(4,10,3)
    train,test=loadData("iris.csv")
    ho=[]
    T=[]
    for i in train:
        x=i[:-1]
        ho.append(f(x,c,b))
        lab=np.zeros(3)
        lab[i[-1]]+=1 #(0,0,0)变成(1,0,0)(0,1,0)(0,0,1)三个类别由0，1区分
        T.append(lab)
    T=np.array(T)
    ho=np.array(ho)
    ho=ho.reshape(130,10)
    hopiv=np.linalg.pinv(ho)
    w=np.dot(hopiv,T)
    w=w.T
    #测试
    ans=[]
    for i in test:
        x=i[:-1]
        hout=f(x,c,b)
        out=np.dot(w,hout)
        if np.argmax(out)==i[-1]:
            ans.append(1)
        else:
            ans.append(0)
    print(sum(ans)/len(ans))
    return sum(ans)/len(ans)

def main():
    acc=[]
    for i in range(10):
        acc.append(Train())
    plt.bar(range(len(acc)),acc)
    plt.show()
    pass
main()
