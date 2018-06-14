import pandas as pda
import numpy as npy
import re
import numpy
from numpy import *

filename = "E:/Software/Python 3 code/5/data/companyall.csv"
dataf = pda.read_csv(filename)
x = dataf.iloc[:,1:7].as_matrix()
y = dataf.iloc[:,7:8].as_matrix()
#处理日期
for i in range(0,len(x)):
    datepat = '^(.*?)\.(.*?)$'
    rst1 = re.compile(datepat).findall(str(x[i][0]))[0]
    year = rst1[0]
    m = rst1[1]
    if(int(year)<2017):
        date1 = (2017-int(year))*12
    else:
        date1 = 0
    if(int(m)>4):
        date2 = 4+12-int(m)-12
    else:
        date2 = 4 - int(m)
    date3 = date1+date2
    if(date3>7):
        #日期这一项就为1
        x[i][0] = int(1)
    else:
        x[i][0]=int(0)
#处理公司规模
    for i in range(0,len(x)):
        try:
            if(int(x[i][1])>15):
                x[i][1] = int(1)
            else:
                x[i][1] = int(0)
        except Exception as err:
            x[i][1] = int(0)
            
