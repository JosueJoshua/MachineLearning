from numpy import *
import operator
from os import listdir
def knn(k,testdata,traindata,labels):
    #testdata:一维数组
    #traindata:二维数组
    #labels:一维列表，和traindata一一对应
    traindatasize = traindata.shape[0]
    dif = tile(testdata,(traindatasize,1)) - traindata
    sqdif = dif ** 2
    sumsqdif = sqdif.sum(axis=1)
    distance = sumsqdif ** 0.5
    sortdistance = distance.argsort()
    count = {}
    for i in range(0,k):
        vote = labels[sortdistance[i]]
        count[vote] = count.get(vote,0) + 1
        sortcount=sorted(count.items(),key=operator.itemgetter(1),reverse=True)
    return sortcount[0][0]

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
#第一种方案
def datatest():
    trainarr,labels = traindata()
    testlist=listdir("E:/Software/Python 3 资料代码/第2次课代码/testdata")
    tnum = len(testlist)
    for i in range(0,tnum):
        thistestfile = testlist[i]
        testarr = datatoarray("E:/Software/Python 3 资料代码/第2次课代码/testdata/"+thistestfile)
        rknn = knn(3,testarr,trainarr,labels)
        print(rknn)
datatest()
#第二种方案
trainarr,labels = traindata()
thistestfile = "6_81.txt"
testarr = datatoarray("E:/Software/Python 3 资料代码/第2次课代码/testdata/" + thistestfile)
rknn = knn(3,testarr,trainarr,labels)
print(rknn)


