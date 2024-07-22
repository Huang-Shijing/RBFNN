import numpy as np
import random
from sklearn.utils import shuffle
import pandas as pd 
import matplotlib.pyplot as plt

def KNN(data,train_num=120,num=10,Flag=0.1):
    Center = [] #初始中心点索引
    center = [] #中心点数据
    result = np.full(train_num,-1)
    #防止初始化质心时重复
    def There_is_no(a):
        for i in Center:
            if a == i:
                return False
        return True
    #随机选定num个初始中心点,坐标保存到center中
    i=0
    while i<num:
        a = random.randint(0,train_num-1)
        if There_is_no(a):
            Center.append(a)
            i+=1
    for i in range(len(Center)):
        center.append(data[Center[i]])
        result[Center[i]]=i
    T=0
    rz_=0
    while True:
        T+=1
        #计算每一个样本的类别
        for i in range(train_num):
            flag = 0
            min = float("inf") #无穷大
            for j in range(len(center)):
                L = np.sqrt(np.sum((data[i]-center[j])**2))#欧氏距离
                if min > L:
                    min = L
                    flag = j
            result[i] = flag
        #聚类算法重新每一类计算中心点：
        center_=np.array(center)
        rz=0
        for i in range(num):
            a = np.zeros(len(data[0]))
            count = 0
            for j in range(train_num):
                if result[j] == i:
                    count+=1
                    a=np.add(a,data[j])
            center[i]=a/count
            rz+= np.sqrt(np.sum((center[i]-center_[i])**2))
        #若选取的新质点位移比较小，不再重新选取
        if abs(rz-rz_)<Flag:
            break
        rz_=rz
        #输出选取的基函数中心点
    return center

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

def init(inputnum,hiddennum,outnum,center,Data):
    c = center
    b = np.full((hiddennum,1) , 1)
    cmax = 0
    for k in range(hiddennum):
        for i in range(len(c)):
            for j in range(len(c)):
                if cmax < np.linalg.norm(c[i]-c[j],2):
                    cmax = np.linalg.norm(c[i]-c[j],2)
    b = (cmax * b) / np.sqrt(2 * hiddennum)#标准差
    return c,b

def f(x,c,b):
    x=c-x
    X=[]
    for i in range(len(x)):
        X.append(-sum(x[i]*x[i])/(2*b[i]*b[i]))
    X=np.matrix(X)
    return np.exp(X)

def Train():
    train,test=loadData("iris.csv")
    knnData=train[:,:-1]
    center = KNN(knnData,120,10,0.001)
    c,b=init(4,10,3,center,train[:,:-1])
    ho=[]
    T=[]
    for i in train:
        x=i[:-1]
        ho.append(f(x,c,b))
        lab=np.zeros(3)
        lab[i[-1]]+=1
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