'''
from sklearn.tree import DecisionTreeClassifier as DTC
dtc = DTC(criterion = entropy)
dtc.fit(x,y)
dtc.predict(x2)
'''
import numpy as npy
import pandas as pda
fname = "E:/Software/Python 3 code/3/lesson.csv"
dataf = pda.read_csv(fname,encoding = "gbk")
x = dataf.iloc[:,1:5].as_matrix()
y = dataf.iloc[:,5].as_matrix()
for i in range(0,len(x)):
    for j in range(0,len(x[i])):
        thisdata = x[i][j]
        if(thisdata == "是" or thisdata == "多"):
            x[i][j] = int(1)
        else:
            x[i][j] = int(0)
for i in range(0,len(y)):
    thisdata = y[i]
    if(thisdata == "高"):
        y[i] = int(1)
    else:
        y[i] = int(0)
#处理里面数据的对象类型
#采用中转法处理：先将不确定类型的数组转为数据框，再将数据框强制类型转换为矩阵数组
xf = pda.DataFrame(x)
yf = pda.DataFrame(y)
x = xf.as_matrix().astype(int)
y = yf.as_matrix().astype(int)

from sklearn.tree import DecisionTreeClassifier as DTC
dtc = DTC(criterion = "entropy")
dtc.fit(x,y)

print(dtc.predict(x))
#计算哪些预测错误，那些预测正确
y = npy.array([i[0] for i in y])
y2 = dtc.predict(x)
print(y == y2)

#可视化决策树
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
with open("E:/Software/Python 3 code/3/tree.dot","w") as file:
    file = export_graphviz(dtc,feature_names = ["shizhan","keshishu","chuxiao","ziliao"],out_file = file)
