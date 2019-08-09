# -*- coding: utf-8 -*-

from sklearn.model_selection import train_test_split
import xgboost as xgb
from gen_feature import make_train_set
from gen_feature import make_test_set
import datetime





def xgboost_make_submission():#这块是得到最终要提交的测试集的结果



    user_listing, training_data, label = make_train_set()
    #划分训练集和验证集，此处的dtest指的应该是验证集
    X_train, X_test, y_train, y_test = train_test_split(training_data.values, label.values, test_size=0.2, random_state=0)
    dtrain=xgb.DMatrix(X_train, label=y_train)
    dtest=xgb.DMatrix(X_test, label=y_test)
    param = {'learning_rate' : 0.1, 'n_estimators': 1000,
             'max_depth': 5, 
        'min_child_weight': 5, 
        'gamma': 0, 
        'subsample': 1.0, 
        'colsample_bytree': 0.8,
        'scale_pos_weight': 1, 
        'eta': 0.05, 
        'silent': 1,
        'objective': 'multi:softmax',
         'num_class':32
       }
    num_round = 283
    param['nthread'] = 4
    #param['eval_metric'] = "auc"
    plst = param.items()
#    plst += [('eval_metric', 'logloss')]
    evallist = [(dtest, 'eval'), (dtrain, 'train')]
    bst=xgb.train(plst, dtrain, num_round, evallist)
    #得到测试集
    sub_user_listing, sub_trainning_data = make_test_set()
    sub_trainning_data = xgb.DMatrix(sub_trainning_data.values)
    y = bst.predict(sub_trainning_data)
    sub_user_listing['label'] = y
    
    
#    pred = sub_user_index[sub_user_index['label'] >= 0]
##    pred = pred[['user_id', 'sku_id']]
#    pred = pred.groupby('user_id').first().reset_index()
#    pred['user_id'] = pred['user_id'].astype(int)
#    pred.to_csv('submission_label_hcq_2.0.csv', index=False, index_label=False)

    return sub_user_listing
if __name__ == '__main__':
    start  =  datetime.datetime.now()

    sub_user_listing = xgboost_make_submission()
    sub_user_listing.to_csv('submission_label_hcq_2.0.csv', index=False, index_label=False)
    end  =  datetime.datetime.now()
    print('xgb训练总共用时',end-start)


