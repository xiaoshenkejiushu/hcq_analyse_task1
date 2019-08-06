# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np


data =  pd.read_csv('D:/code/pythonfile/datawhale/data.csv',encoding = 'gbk')

print(data.columns)

#1数据类型的分析
#for col in data.columns:
#    print(col)
#    print(type(data[col][2]))


#无关特征的删除
#问题，如何鉴定无关特征，1.某列的数值类别单一2.一些id特征 3看数据分布
#for col in data.columns:
#    print(col)
#    print(len(pd.unique(data[col])))

#删掉的列：bank_card_no1、student_feature3、source1
print(len(data.columns))
data = data.drop(['bank_card_no','student_feature','source'],axis =1)
print(len(data.columns))







