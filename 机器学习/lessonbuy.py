from apriori import *
import pandas as pda
filename = "E:/Software/Python 3 code/6/lesson_buy.xls"
dframe = pda.read_excel(filename,header = None)
change = lambda x:pda.Series(1,index = x[pda.notnull(x)])
mapok = map(change,dframe.as_matrix())
data = pda.DataFrame(list(mapok)).fillna(0)
support = 0.2
cfd = 0.3
print(find_rule(data,support,cfd))
