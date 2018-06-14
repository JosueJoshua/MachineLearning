import pandas as pda
fname = "E:/Software/Python 3 code/5/data/luqu.csv"
dataf = pda.read_csv(fname)
x = dataf.iloc[:,1:4].as_matrix()
y = dataf.iloc[:,0:1].as_matrix()
from sklearn.linear_model import LogisticRegression as LR

r1 = LR()
r1.fit(x,y)
print("训练结束")
print("模型正确率为：" + str(r1.score(x,y)))
rst = r1.predict(x)
