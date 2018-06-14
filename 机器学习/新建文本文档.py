#加载datasets里面的数据
from sklearn import datasets
irisdata = datasets.load_iris()
#特征（data）、类别（target）
x = irisdata.data
y = irisdata.target
#加载第三方数据
import numpy as npy
import pandas as pda
path = "E:/Software/Python 3 code/8/luqu.csv"
dataf = pda.read_csv(path)
x2 = dataf.iloc[:,1:4].as_matrix()
y2 = dataf.iloc[:,0:1].as_matrix()
#如何数据集的切分
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 20)
#数据归一化
from sklearn import preprocessing
nx = preprocessing.normalize(x)
#数据标准化
sx = preprocessing.scale(x)
#特征筛选
from sklearn.ensemble import ExtraTreesClassifier
model = ExtraTreesClassifier()
model.fit(x,y)
rst = model.feature_importances_
#print(rst)
#常见算法的实现
#常见算法的实现-k近邻
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()
model.fit(x2,y2)
tx2 = npy.array([[800,3.50,1],[372,3.71,2]])
print(model.predict(tx2))
