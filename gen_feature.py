# -*- coding: utf-8 -*-
import pandas as pd 
import numpy as np
import datetime


dir_path = 'D:/code2019/o2o_tianchi/data/'

online_test = pd.read_csv(dir_path+'ccf_offline_stage1_test_revised .csv')

offline_train = pd.read_csv(dir_path+'ccf_offline_stage1_train .csv')

online_train = pd.read_csv(dir_path+'ccf_online_stage1_train .csv')

online_train = pd.read_csv(dir_path+'ccf_online_stage1_train .csv')






#1.01  产生一些基础的特征
def get_basic_train_fea():
    start  =  datetime.datetime.now()
    train_ori = ori_train.sort_values(by = ['user_id','listing_id'])
    train_ori['repay_date'] = train_ori['repay_date'].replace("\\N",'2020-01-01')
    train_ori['repay_amt'] = train_ori['repay_amt'].replace("\\N",0)
    
    #将时间都由字符串形式转换成可以识别的datetime形式
    train_ori['auditing_date'] = pd.to_datetime(train_ori['auditing_date'])
    train_ori['due_date'] = pd.to_datetime(train_ori['due_date'])
    train_ori['repay_date'] = pd.to_datetime(train_ori['repay_date'])
    #将时间做差，算出间隔时间
    train_ori['time_delt_repay_due'] =  train_ori['repay_date'] - train_ori['due_date']
    train_ori['time_delt_repay_due'] = train_ori['time_delt_repay_due'].dt.days
    
    train_ori = train_ori[['user_id','listing_id','time_delt_repay_due','due_amt']]
#    train_ori  =  train_ori.groupby(by=['user_id'],as_index =False).mean()


    end  =  datetime.datetime.now()
    print('train总共用时',end-start)
    return train_ori

def get_basic_test_fea():
    
    test_ori = ori_test.sort_values(by = ['user_id'])
    test_ori['auditing_date'] = pd.to_datetime(test_ori['auditing_date'])
    test_ori['due_date'] = pd.to_datetime(test_ori['due_date'])
    test_ori = test_ori[['user_id','listing_id','due_amt']]
    
    return test_ori

def get_basic_listing_fea():
    start  =  datetime.datetime.now()
    
    listing_info_copy_1 = ori_listing_info.sort_values(by = ['user_id','listing_id'])
    listing_info_copy_2 = ori_listing_info.sort_values(by = ['user_id','listing_id'])
    
    listing_info_copy_1 = listing_info_copy_1[['user_id','term','rate','principal']]
    listing_info_copy_1 = listing_info_copy_1.groupby(by=['user_id'],as_index= False).mean()
    
    listing_info_copy_2['owe_quantity'] = int(1)
    listing_info_copy_2 = listing_info_copy_2[['user_id','owe_quantity']]
    listing_info_copy_2 = listing_info_copy_2.groupby(by=['user_id'],as_index = False).sum()
    
    listing_info_copy_1['owe_quantity'] = listing_info_copy_2['owe_quantity']
    end  =  datetime.datetime.now()
    print('listing总共用时',end-start)
    return listing_info_copy_1

def get_basic_user_fea():
    
    start  =  datetime.datetime.now()
    
    user_ori = ori_user_info.sort_values(by = ['user_id'])
    user_ori['reg_mon'] = pd.to_datetime(user_ori['reg_mon'])
    user_ori['now_reg_mon'] = start - user_ori['reg_mon']
    user_ori['now_reg_day'] = user_ori['now_reg_mon'].dt.days
    
    user_ori = user_ori[['user_id','age','now_reg_day']]
    end  =  datetime.datetime.now()
    print('user总共用时',end-start)
    return user_ori

#1.02  构造训练集、测试集、验证集
    

def make_train_set():
    
    train_ori =  get_basic_train_fea()
    listing_ori = get_basic_listing_fea()
    user_ori = get_basic_user_fea()
    
    train_paipai = pd.merge(train_ori,listing_ori,how = 'left', on = ['user_id'])
    train_paipai = pd.merge(train_paipai,user_ori,how =  'left', on = ['user_id'])
    train_paipai =train_paipai.fillna(0)
    train_paipai[train_paipai['time_delt_repay_due']>0] = 31 
    train_paipai[train_paipai['time_delt_repay_due']<-30] = -30 
    train_paipai['time_delt_repay_due'] = train_paipai['time_delt_repay_due'].abs()
    
    user_listing = train_paipai[['user_id', 'listing_id']].copy()
    labels = train_paipai['time_delt_repay_due'].copy()
    del train_paipai['user_id']
    del train_paipai['listing_id']
    del train_paipai['time_delt_repay_due']
    
    return user_listing, train_paipai, labels

def make_test_set():
    test_ori = get_basic_test_fea()
    listing_ori = get_basic_listing_fea()
    user_ori = get_basic_user_fea()
    test_paipai = pd.merge(test_ori,listing_ori,how = 'left', on = ['user_id'])
    test_paipai = pd.merge(test_paipai,user_ori,how =  'left', on = ['user_id'])
    test_paipai =test_paipai.fillna(0)    
    
    user_listing = test_paipai[['user_id', 'listing_id']].copy()
    del test_paipai['user_id']
    del test_paipai['listing_id']
    
    return user_listing, test_paipai

if __name__ == '__main__':
    #train_ori =  get_basic_train_fea()
    #train_ori['time_delt_repay_due'] = train_ori['time_delt_repay_due'].apply(lambda x:np.int(x))
    #listing_ori = get_basic_listing_fea()
    
    #可以在这个界面上做，做完再合过去
    user_listing_train, train_paipai, labels = make_train_set()
    user_listing_test, test_paipai = make_test_set()
    df_train = pd.concat([user_listing_train,train_paipai],axis=1)
    df_test = pd.concat([user_listing_test,test_paipai],axis=1)
    df_total = pd.concat([df_train,df_test],axis=0,ignore_index = True)
    
    df_total = df_total.sort_values(by='now_reg_day', ascending=False).drop_duplicates('user_id').reset_index(drop=True)
    df_total = df_total[['user_id','owe_quantity','now_reg_day']]
    









