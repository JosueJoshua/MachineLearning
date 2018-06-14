from numpy import *
from math import log
from math import e

def dcstumpclass(data,i,k,thresh):
    a,b = shape(data)
    classrst = ones((a,1))
    if k == 0:
        classrst[data[:,i] <= thresh] = -1
    elif k == 1:
        classrst[data[:,i] > thresh] = -1
    return classrst
def dcstump(data,labels):
    a,b=shape(data)
    D = ones((a,1))/a
    allstep = 10  #a数据条数,b特征数
    minerror = inf  #正无穷
    for i in range(0,b):
        max1 = data[:,i].max()
        min1 = data[:,i].min()
        step = (max1 - min1) / allstep
        for j in range(-1,int(allstep)+1):
            thresh = min1 + j*step  #计算当前阈值
            #进入类别循环
            for k in range(0,2):
                classrst = dcstumpclass(data,i,k,thresh)
                errorarr = ones((a,1))
                errorarr[classrst[:,0] == labels] = 0
                errorarr = mat(errorarr)
                weighterror = D.T*errorarr
                if(weighterror < minerror):
                    minerror = weighterror
                    besti = i
                    bestthresh = thresh
                    bestk = k
                    bestclass = classrst.copy()
    return minerror,besti,bestthresh,bestk,bestclass
data = array([[1.8,1.2],
             [1.3,1.7],
             [2.5,1.6],
             [2.9,1.1]])
labels = [1.0,1.0,-1.0,-1.0]
best = dcstump(data,labels)
print(best)
data2 = array([[1.8,1.3],[2.8,1.1]])
rst = dcstumpclass(data2,0,1,1.9399999999999999)
print(rst)
