from numpy import *
import operator
from os import listdir
#从列方向扩展
#tile(a,(size,1))
def knn(k,testdata,traindata,labels):
    traindatasize=traindata.shape[0]
    dif=tile(testdata,(traindatasize,1))-traindata
    sqdif=dif**2
    sumsqdif=sqdif.sum(axis=1)
    distance=sumsqdif**0.5
    sortdistance=distance.argsort()
    count={}
    for i in range(0,k):
        vote=labels[sortdistance[i]]
        count[vote]=count.get(vote,0)+1
    sortcount=sorted(count.items(),key=operator.itemgetter(1),reverse=True)
    return sortcount[0][0]


#图片处理
#先将所有图片转为固定宽高，比如32*32，然后再转为文本

#加载数据
def datatoarray(fname):
    arr=[]
    fh=open(fname)
    for i in range(0,32):
        thisline=fh.readline()
        for j in range(0,32):
            arr.append(int(thisline[j]))
    return arr
arr1=datatoarray("D:/Python35/traindata/0_4.txt")
#建立一个函数取文件名前缀
def seplabel(fname):
    filestr=fname.split(".")[0]
    label=int(filestr.split("_")[0])
    return label
#建立训练数据
def traindata():
    labels=[]
    trainfile=listdir("D:/Python35/traindata")
    num=len(trainfile)
    #长度1024（列），每一行存储一个文件
    #用一个数组存储所有训练数据，行：文件总数，列：1024
    trainarr=zeros((num,1024))
    for i in range(0,num):
        thisfname=trainfile[i]
        thislabel=seplabel(thisfname)
        labels.append(thislabel)
        trainarr[i,:]=datatoarray("traindata/"+thisfname)
    return trainarr,labels
#用测试数据调用KNN算法去测试，看是否能够准确识别
def datatest():
    trainarr,labels=traindata()
    testlist=listdir("testdata")
    tnum=len(testlist)
    for i in range(0,tnum):
        thistestfile=testlist[i]
        testarr=datatoarray("testdata/"+thistestfile)
        rknn=knn(3,testarr,trainarr,labels)
        print(rknn)
#datatest()
#抽某一个测试文件出来进行试验
trainarr,labels=traindata()
thistestfile="8_76.txt"
testarr=datatoarray("testdata/"+thistestfile)
rknn=knn(3,testarr,trainarr,labels)
print(rknn)
