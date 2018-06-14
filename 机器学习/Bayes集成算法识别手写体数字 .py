#贝叶斯算法
from collections import Counter

import numpy as npy
import numpy
from numpy import *
import operator
from os import listdir

from sklearn.naive_bayes import GaussianNB
from sklearn import datasets

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
        testarr = datatoarray("E:/Software/Python 3 资料代码/第2次课代码/testdata/"+thistestfile)
        test.append(testarr)
    return test,testlabels

trainarr,labels = traindata()
test,testlabels = datatest()
model = GaussianNB()
model.fit(trainarr,labels)
expected = testlabels
predicted = model.predict(test)
#计算准确率
judge = (npy.array(predicted) == testlabels)
c = Counter(judge)
accuracy = c[True] / len(testlabels)
