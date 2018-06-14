import math
#计算log以2为底X的对数
def getlog2(x):
    a = math.log(x,2)
    return a
#计算信息熵
def getI(x1,x2):
    p1 = x1 / (x1 + x2)
    p2 = x2 / (x1 + x2)
    i =- p1*getlog2(p1)-p2*getlog2(p2)
    return i
#计算e值
def gete(x1,x2,x3,x4):
    p1 = (x1 + x2)/(x1+x2+x3+x4)
    p2 = (x3 + x4)/(x1+x2+x3+x4)
    i1 = getI(x1,x2)
    i2 = getI(x3,x4)
    e = p1 * i1 +p2*i2
    return e
#计算价格这个特征的e值
pricee = gete(2,1,2,2)
#计算课时数这个特征的e值
nume = gete(2,0,2,3)
#计算信息增益
def getgrain(x1,x2,x3,x4,x5,x6):
    g = getI(x5,x6)-gete(x1,x2,x3,x4)
    return g
#计算价格的信息增益
pricegrain = getgrain(2,1,2,2,4,3)
