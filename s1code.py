# -*- coding: utf-8 -*-

import os, sys, pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date
from sklearn.linear_model import SGDClassifier, LogisticRegression
#采用逻辑回归和随机梯度下降来做这个比赛。


dir_path = 'E:/code2019/o2o_tianchi/data/'

dftest = pd.read_csv(dir_path+'ccf_offline_stage1_test_revised .csv')

dfoff = pd.read_csv(dir_path+'ccf_offline_stage1_train.csv')

dfon = pd.read_csv(dir_path+'ccf_online_stage1_train.csv')

#online_train = pd.read_csv(dir_path+'ccf_online_stage1_train.csv')


# 1. 将满xx减yy类型(`xx:yy`)的券变成折扣率 : `1 - yy/xx`，同时建立折扣券相关的特征 `discount_rate, discount_man, discount_jian, discount_type`
# 2. 将距离 `str` 转为 `int`
# convert Discount_rate and Distance
def getDiscountType(row):#有引号的设为1，没引号的设为0
    if pd.isnull(row):
        return np.nan
    elif ':' in row:
        return 1
    else:
        return 0
    

def convertRate(row):#将满减转化成折扣率
    """Convert discount to rate"""
    if pd.isnull(row):#填充缺失值
        return 1.0
    elif ':' in str(row):#我那里用的是正则，它这里用的是split
        rows = row.split(':')
        return 1.0 - float(rows[1])/float(rows[0])
    else:
        return float(row)

def getDiscountMan(row):#把满作为特征返回
    if ':' in str(row):
        rows = row.split(':')
        return int(rows[0])
    else:
        return 0

def getDiscountJian(row):#把减作为特征返回
    if ':' in str(row):
        rows = row.split(':')
        return int(rows[1])
    else:
        return 0
print("tool is ok.")


def processData(df):#用apply函数制作4个相关特征
    # convert discunt_rate
    df['discount_rate'] = df['Discount_rate'].apply(convertRate)
    df['discount_man'] = df['Discount_rate'].apply(getDiscountMan)
    df['discount_jian'] = df['Discount_rate'].apply(getDiscountJian)
    df['discount_type'] = df['Discount_rate'].apply(getDiscountType)
    print('discount_rate',df['discount_rate'].unique())
    # convert distance
    df['distance'] = df['Distance'].fillna(-1).astype(int)
    return df

dfoff = processData(dfoff)
dftest = processData(dftest)#对训练集和测试集分别进行特征处理

date_received = dfoff['Date_received'].unique()
date_received = sorted(date_received[pd.notnull(date_received)])#对接收列的非空值进行了排序

date_buy = dfoff['Date'].unique()
date_buy = sorted(date_buy[pd.notnull(date_buy)])#对购买列的非空值进行了排序
date_buy = sorted(dfoff[dfoff['Date'].notnull()]['Date'])#这个跟上面的排序是一样的，只不过换了种表达方式
couponbydate = dfoff[dfoff['Date_received'].notnull()][['Date_received', 'Date']].groupby(['Date_received'], as_index=False).count()#接受列非空接收日期的个数
couponbydate.columns = ['Date_received','count']
buybydate = dfoff[(dfoff['Date'].notnull()) & (dfoff['Date_received'].notnull())][['Date_received', 'Date']].groupby(['Date_received'], as_index=False).count()#购买者非空的个数
buybydate.columns = ['Date_received','count']
#上面的部分其实算出了两个数，第一个是有多少接收优惠券的天，第二个是有多少使用优惠券的天,基本上这俩列是
print(buybydate['count']-couponbydate['count'])
'''为嘛全是零，有点难以理解'''
print(couponbydate)

print("end")


def getWeekday(row):#这是一个返回它是周几的函数
    if row == 'nan':
        return np.nan
    else:
        return date(int(row[0:4]), int(row[4:6]), int(row[6:8])).weekday() + 1

dfoff['weekday'] = dfoff['Date_received'].astype(str).apply(getWeekday)
dftest['weekday'] = dftest['Date_received'].astype(str).apply(getWeekday)

# weekday_type :  周六和周日为1，其他为0
dfoff['weekday_type'] = dfoff['weekday'].apply(lambda x : 1 if x in [6,7] else 0 )
dftest['weekday_type'] = dftest['weekday'].apply(lambda x : 1 if x in [6,7] else 0 )


#将新生成的weekday列对它进行哑编码,对test和train两列分别进行了这个处理
weekdaycols = ['weekday_'+str(i) for i in range(1,8)]
tmp_df = pd.get_dummies(dfoff['weekday'].replace('nan',np.nan))
tmp_df.columns = weekdaycols
dfoff[weekdaycols]= tmp_df


tmpdf = pd.get_dummies(dftest['weekday'].replace('nan', np.nan))
tmpdf.columns = weekdaycols
dftest[weekdaycols] = tmpdf


#制作标签列
def label(row):
    if pd.isnull(row['Date_received']):
        return -1
    if pd.notnull(row['Date']):
        td = pd.to_datetime(row['Date'], format='%Y%m%d') -  pd.to_datetime(row['Date_received'], format='%Y%m%d')
        if td <= pd.Timedelta(15, 'D'):
            return 1
    return 0
dfoff['label'] = dfoff.apply(label, axis = 1)

print("end")


# data split
print("-----data split------")
df = dfoff[dfoff['label']!=-1].copy()
train = df[df['Date_received']<20160516].copy()
valid = df[(df['Date_received']>20160516)&(df['Date_received']<20160615)].copy()
print("end")


original_feature = ['discount_rate','discount_type','discount_man', 'discount_jian','distance', 'weekday', 'weekday_type'] + weekdaycols
print('------train--------')
model = SGDClassifier(#lambda:
    loss='log',
    penalty='elasticnet',
    fit_intercept=True,
    max_iter=100,
    shuffle=True,
    alpha = 0.01,
    l1_ratio = 0.01,
    n_jobs=1,
    class_weight=None
)
model.fit(train[original_feature], train['label'])
score = model.score(valid[original_feature], valid['label'])
print('验证集得分',score)

print("---save model---")
with open('sgd_model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('sgd_model.pkl', 'rb') as f:
    model = pickle.load(f)
    
y_test_pred = model.predict_proba(dftest[original_feature])
dftest1 = dftest[['User_id','Coupon_id','Date_received']].copy()
dftest1['label'] = y_test_pred[:,1]
dftest1.to_csv('submit1.csv', index=False, header=False)
dftest1.head()





