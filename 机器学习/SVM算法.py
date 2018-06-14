import numpy as npy
from sklearn import svm
import matplotlib.pyplot as plt
x1 = []  #存储对应数据
y1 = []  #存储对应数据类别
for i in range(0,10):
    if(i <= 3 or i>=8):
        x1.append([i,i])
        y1.append(0)
    else:
        x1.append([i,i])
        y1.append(1)
x = npy.array(x1)
y = npy.array(y1)
'''创建SVM支持向量机'''
#svc模型 = svm.SVC(kernel = 核函数).fit(训练数据,训练类别)
#线性核函数
linear = svm.SVC(kernel = 'linear').fit(x,y)

#多项式核函数
poly = svm.SVC(kernel = 'poly',degree = 4).fit(x,y)

#径向基核函数
rbf = svm.SVC().fit(x,y)

#Sigmoid核函数
sigmoid = svm.SVC(kernel = 'sigmoid').fit(x,y)
x21,x22 = npy.meshgrid(npy.arange(x[:,0].min(),x[:,0].max(),0.01),npy.arange(x[:,1].min(),x[:,1].max(),0.01))
a=1
for i in [linear,poly,rbf,sigmoid]:
    rst = i.predict(npy.c_[x21.ravel(),x22.ravel()])
    #plt.subplot(横向划分,纵向划分，定位)
    plt.subplot(2,2,a)
    plt.contourf(x21,x22,rst.reshape(x21.shape))
    for j in range(0,len(y1)):
        if(int(y1[j]) == 0):
            plt.plot(x[j:j+1,0],x[j:j+1,1],"yo")
        else:
            plt.plot(x[j:j+1,0],x[j:j+1,1],"ko")
    a+=1
plt.show()
