#贝叶斯原生算法
from collections import Counter

import numpy as npy
import numpy
from numpy import *
import operator
from os import listdir
class Bayes:
    def __init__(self):
        self.length=-1
        self.labelcount=dict()#各类别的概率{"类别1"：概率1，"类别2"：概率2,…}
        self.vectorcount=dict()#以字典的方式存储类别与特征向量，
        #格式为{"类别1":[特征向量1,特征向量2,…],…,"类别n":[特征向量1,特征向量2,…]}
    def fit(self,dataSet:list,labels:list):
        if(len(dataSet)!=len(labels)):
            raise ValueError("您输入的训练数组跟类别数组长度不一致")
        self.length=len(dataSet[0])#训练数据特征值的长度
        labelsnum=len(labels)#类别所有的数量（可重复）
        norlabels=set(labels)#不重复类别的数量
        for item in norlabels:
            thislabel=item
            self.labelcount[thislabel]=labels.count(thislabel)/labelsnum#求的当前类别占类别总数的比例，p(c)
        #通过zip将两个数组交叉放置，比如：


        for vector,label in zip(dataSet,labels):
            if(label not in self.vectorcount):
                self.vectorcount[label]=[]
            self.vectorcount[label].append(vector)
        print(self.labelcount)
        print("训练结束")
        return self
    def btest(self,TestData,labelsSet):
        if(self.length==-1):
            raise ValueError("您还没有进行训练，请先训练")
        #计算testdata分别为各个类别的概率
        lbDict=dict()#{"类别1"：概率1，“类别2”：概率2}
        for thislb in labelsSet:
            p=1
            alllabel=self.labelcount[thislb]#当前类别的概率p(c)
            allvector=self.vectorcount[thislb]#当前类别的所有特征向量
            vnum=len(allvector)#当前类别特征向量个数
            allvector=numpy.array(allvector).T#转置一下

            for index in range(0,len(TestData)):#依次计算各特征的概率
                vector=list(allvector[index])
                p*=vector.count(TestData[index])/vnum#p(当前特征|C)
                #如testdata[0]:0,vector:[1,0,1,0,0,1,1],p为3/7
            lbDict[thislb]=p*alllabel#alllabel相当于p(c)
        thislabel=sorted(lbDict,key=lambda x:lbDict[x],reverse=True)[0]
        return thislabel
#----------------------------------------------------------------------------

#加载数据
def datatoarray(file):
    arr=[]
    fl = open(file)
    for i in range(0,32):
        thisline = fl.readline()
        for j in range(0,32):
            arr.append(int(thisline[j]))
    return arr
#arr1 = dataarray("E:/Software/Python 3 资料代码/第2次课代码/traindata/0_0.txt")

def seplabel(fname):
    filestr = fname.split(".")[0]
    label = int(filestr.split("_")[0])
    return label

def traindata():
    labels = []
    trainfile = listdir("E:/Software/Python 3 资料代码/第2次课代码/traindata")
    num = len(trainfile)
    trainarr = zeros((num,1024))
    for i in range(0,num):
        thisfname = trainfile[i]
        thislabel = seplabel(thisfname)
        labels.append(thislabel)
        trainarr[i,:] = datatoarray("E:/Software/Python 3 资料代码/第2次课代码/traindata/" + thisfname)
    return trainarr,labels
#----------------------------------------------------------------------------

def datatest():
    testlist=listdir("E:/Software/Python 3 资料代码/第2次课代码/testdata")
    tnum = len(testlist)
    testlabels = []
    test = []
    for i in range(0,tnum):
        thistestfile = testlist[i]
        thislabel = seplabel(thistestfile)
        testlabels.append(thislabel);
    for j in range(0,tnum):
        thistestfile = testlist[j]
        testarr = datatoarray("E:/Software/Python 3 资料代码/第2次课代码/testdata/"+thistestfile)
        predict = bys.btest(testarr,set(testlabels))
        test.append(predict)
        print(predict)
    print(test)
    return test,testlabels

trainarr,labels = traindata()
bys = Bayes()
bys.fit(trainarr,labels)
testpredict,testlabel = datatest()
#计算准确率
judge = (npy.array(testpredict) == testlabel)
c = Counter(judge)
accuracy = c[True] / len(testlabel)

