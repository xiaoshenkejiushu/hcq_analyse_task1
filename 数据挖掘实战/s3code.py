# -*- coding: utf-8 -*-
'''
using：复现（天池 o2o 参考代码）
link：https://tianchi.aliyun.com/notebook-ai/detail?postId=23504
auc：
'''




import datetime
import os
import time
from concurrent.futures import ProcessPoolExecutor
from math import ceil

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from xgboost.sklearn import XGBClassifier
from function1 import *


pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 200)



    
    



def grid_search_gbdt(get_param=False):
    params = {
        # 10
        'learning_rate': 1e-2,
        'n_estimators': 1900,
        'max_depth': 9,
        'min_samples_split': 200,
        'min_samples_leaf': 50,
        'subsample': .8,

        # 'learning_rate': 1e-1,
        # 'n_estimators': 200,
        # 'max_depth': 8,
        # 'min_samples_split': 200,
        # 'min_samples_leaf': 50,
        # 'subsample': .8,

        'random_state': 0
    }

    if get_param:
        return params

    steps = {
        'n_estimators': {'step': 100, 'min': 1, 'max': 'inf'},
        'max_depth': {'step': 1, 'min': 1, 'max': 'inf'},
        'min_samples_split': {'step': 10, 'min': 2, 'max': 'inf'},
        'min_samples_leaf': {'step': 10, 'min': 1, 'max': 'inf'},
        'subsample': {'step': .1, 'min': .1, 'max': 1},
    }

    grid_search_auto(steps, params, GradientBoostingClassifier())


def grid_search_xgb(get_param=False):
    params = {
        # all
        # 'learning_rate': 1e-1,
        # 'n_estimators': 80,
        # 'max_depth': 8,
        # 'min_child_weight': 3,
        # 'gamma': .2,
        # 'subsample': .8,
        # 'colsample_bytree': .8,

        # 10
        'learning_rate': 1e-2,
        'n_estimators': 1260,
        'max_depth': 8,
        'min_child_weight': 4,
        'gamma': .2,
        'subsample': .6,
        'colsample_bytree': .8,
        'scale_pos_weight': 1,
        'reg_alpha': 0,

        # 'learning_rate': 1e-1,
        # 'n_estimators': 80,
        # 'max_depth': 8,
        # 'min_child_weight': 3,
        # 'gamma': .2,
        # 'subsample': .8,
        # 'colsample_bytree': .8,
        # 'scale_pos_weight': 1,
        # 'reg_alpha': 0,

        'n_jobs': cpu_jobs,
        'seed': 0
    }

    if get_param:
        return params

    steps = {
        'n_estimators': {'step': 10, 'min': 1, 'max': 'inf'},
        'max_depth': {'step': 1, 'min': 1, 'max': 'inf'},
        'min_child_weight': {'step': 1, 'min': 1, 'max': 'inf'},
        'gamma': {'step': .1, 'min': 0, 'max': 1},
        'subsample': {'step': .1, 'min': .1, 'max': 1},
        'colsample_bytree': {'step': .1, 'min': .1, 'max': 1},
        'scale_pos_weight': {'step': 1, 'min': 1, 'max': 10},
        'reg_alpha': {'step': .1, 'min': 0, 'max': 1},
    }

    grid_search_auto(steps, params, XGBClassifier())


def grid_search_lgb(get_param=False):
    params = {
        # 10
        'learning_rate': 1e-2,
        'n_estimators': 1200,
        'num_leaves': 51,
        'min_split_gain': 0,
        'min_child_weight': 1e-3,
        'min_child_samples': 22,
        'subsample': .8,
        'colsample_bytree': .8,

        # 'learning_rate': .1,
        # 'n_estimators': 90,
        # 'num_leaves': 50,
        # 'min_split_gain': 0,
        # 'min_child_weight': 1e-3,
        # 'min_child_samples': 21,
        # 'subsample': .8,
        # 'colsample_bytree': .8,

        'n_jobs': cpu_jobs,
        'random_state': 0
    }

    if get_param:
        return params

    steps = {
        'n_estimators': {'step': 100, 'min': 1, 'max': 'inf'},
        'num_leaves': {'step': 1, 'min': 1, 'max': 'inf'},
        'min_split_gain': {'step': .1, 'min': 0, 'max': 1},
        'min_child_weight': {'step': 1e-3, 'min': 1e-3, 'max': 'inf'},
        'min_child_samples': {'step': 1, 'min': 1, 'max': 'inf'},
        # 'subsample': {'step': .1, 'min': .1, 'max': 1},
        'colsample_bytree': {'step': .1, 'min': .1, 'max': 1},
    }

    grid_search_auto(steps, params, LGBMClassifier())


def grid_search_cat(get_param=False):
    params = {
        # 10
        'learning_rate': 1e-2,
        'n_estimators': 3600,
        'max_depth': 8,
        'max_bin': 127,
        'reg_lambda': 2,
        'subsample': .7,

        # 'learning_rate': 1e-1,
        # 'iterations': 460,
        # 'depth': 8,
        # 'l2_leaf_reg': 8,
        # 'border_count': 37,

        # 'ctr_border_count': 16,
        'one_hot_max_size': 2,
        'bootstrap_type': 'Bernoulli',
        'leaf_estimation_method': 'Newton',
        'random_state': 0,
        'verbose': False,
        'eval_metric': 'AUC',
        'thread_count': cpu_jobs
    }

    if get_param:
        return params

    steps = {
        'n_estimators': {'step': 100, 'min': 1, 'max': 'inf'},
        'max_depth': {'step': 1, 'min': 1, 'max': 'inf'},
        'max_bin': {'step': 1, 'min': 1, 'max': 255},
        'reg_lambda': {'step': 1, 'min': 0, 'max': 'inf'},
        'subsample': {'step': .1, 'min': .1, 'max': 1},
        'one_hot_max_size': {'step': 1, 'min': 0, 'max': 255},
    }

    grid_search_auto(steps, params, CatBoostClassifier())


def grid_search_rf(criterion='gini', get_param=False):
    if criterion == 'gini':
        params = {
            # 10
            'n_estimators': 3090,
            'max_depth': 15,
            'min_samples_split': 2,
            'min_samples_leaf': 1,

            'criterion': 'gini',
            'random_state': 0
        }
    else:
        params = {
            'n_estimators': 3110,
            'max_depth': 13,
            'min_samples_split': 70,
            'min_samples_leaf': 10,
            'criterion': 'entropy',
            'random_state': 0
        }

    if get_param:
        return params

    steps = {
        'n_estimators': {'step': 10, 'min': 1, 'max': 'inf'},
        'max_depth': {'step': 1, 'min': 1, 'max': 'inf'},
        'min_samples_split': {'step': 2, 'min': 2, 'max': 'inf'},
        'min_samples_leaf': {'step': 2, 'min': 1, 'max': 'inf'},
    }

    grid_search_auto(steps, params, RandomForestClassifier())


def grid_search_et(criterion='gini', get_param=False):
    if criterion == 'gini':
        params = {
            # 10
            'n_estimators': 3060,
            'max_depth': 22,
            'min_samples_split': 12,
            'min_samples_leaf': 1,

            'criterion': 'gini',
            'random_state': 0,
        }
    else:
        params = {
            'n_estimators': 3100,
            'max_depth': 13,
            'min_samples_split': 70,
            'min_samples_leaf': 10,
            'criterion': 'entropy',
            'random_state': 0
        }

    if get_param:
        return params

    steps = {
        'n_estimators': {'step': 10, 'min': 1, 'max': 'inf'},
        'max_depth': {'step': 1, 'min': 1, 'max': 'inf'},
        'min_samples_split': {'step': 2, 'min': 2, 'max': 'inf'},
        'min_samples_leaf': {'step': 2, 'min': 1, 'max': 'inf'},
    }

    grid_search_auto(steps, params, ExtraTreesClassifier())


def train_gbdt(model=False):
    global log

    params = grid_search_gbdt(True)
    clf = GradientBoostingClassifier().set_params(**params)

    if model:
        return clf

    params = clf.get_params()
    log += 'gbdt'
    log += ', learning_rate: %.3f' % params['learning_rate']
    log += ', n_estimators: %d' % params['n_estimators']
    log += ', max_depth: %d' % params['max_depth']
    log += ', min_samples_split: %d' % params['min_samples_split']
    log += ', min_samples_leaf: %d' % params['min_samples_leaf']
    log += ', subsample: %.1f' % params['subsample']
    log += '\n\n'

    return train(clf)


def train_xgb(model=False):
    global log

    params = grid_search_xgb(True)

    clf = XGBClassifier().set_params(**params)

    if model:
        return clf

    params = clf.get_params()
    log += 'xgb'
    log += ', learning_rate: %.3f' % params['learning_rate']
    log += ', n_estimators: %d' % params['n_estimators']
    log += ', max_depth: %d' % params['max_depth']
    log += ', min_child_weight: %d' % params['min_child_weight']
    log += ', gamma: %.1f' % params['gamma']
    log += ', subsample: %.1f' % params['subsample']
    log += ', colsample_bytree: %.1f' % params['colsample_bytree']
    log += '\n\n'

    return train(clf)


def train_lgb(model=False):
    global log

    params = grid_search_lgb(True)

    clf = LGBMClassifier().set_params(**params)

    if model:
        return clf

    params = clf.get_params()
    log += 'lgb'
    log += ', learning_rate: %.3f' % params['learning_rate']
    log += ', n_estimators: %d' % params['n_estimators']
    log += ', num_leaves: %d' % params['num_leaves']
    log += ', min_split_gain: %.1f' % params['min_split_gain']
    log += ', min_child_weight: %.4f' % params['min_child_weight']
    log += ', min_child_samples: %d' % params['min_child_samples']
    log += ', subsample: %.1f' % params['subsample']
    log += ', colsample_bytree: %.1f' % params['colsample_bytree']
    log += '\n\n'

    return train(clf)


def train_cat(model=False):
    global log

    params = grid_search_cat(True)

    clf = CatBoostClassifier().set_params(**params)

    if model:
        return clf

    params = clf.get_params()
    log += 'cat'
    log += ', learning_rate: %.3f' % params['learning_rate']
    log += ', iterations: %d' % params['iterations']
    log += ', depth: %d' % params['depth']
    log += ', l2_leaf_reg: %d' % params['l2_leaf_reg']
    log += ', border_count: %d' % params['border_count']
    log += ', subsample: %d' % params['subsample']
    log += ', one_hot_max_size: %d' % params['one_hot_max_size']
    log += '\n\n'

    return train(clf)


def train_rf(clf):
    global log

    params = clf.get_params()
    log += 'rf'
    log += ', n_estimators: %d' % params['n_estimators']
    log += ', max_depth: %d' % params['max_depth']
    log += ', min_samples_split: %d' % params['min_samples_split']
    log += ', min_samples_leaf: %d' % params['min_samples_leaf']
    log += ', criterion: %s' % params['criterion']
    log += '\n\n'

    return train(clf)


def train_rf_gini(model=False):
    clf = RandomForestClassifier().set_params(**grid_search_rf('gini', True))
    if model:
        return clf
    return train_rf(clf)


def train_rf_entropy():
    clf = RandomForestClassifier().set_params(**grid_search_rf('entropy', True))

    return train_rf(clf)


def train_et(clf):
    global log

    params = clf.get_params()
    log += 'et'
    log += ', n_estimators: %d' % params['n_estimators']
    log += ', max_depth: %d' % params['max_depth']
    log += ', min_samples_split: %d' % params['min_samples_split']
    log += ', min_samples_leaf: %d' % params['min_samples_leaf']
    log += ', criterion: %s' % params['criterion']
    log += '\n\n'

    return train(clf)


def train_et_gini(model=False):
    clf = ExtraTreesClassifier().set_params(**grid_search_et('gini', True))
    if model:
        return clf
    return train_et(clf)


def train_et_entropy():
    clf = ExtraTreesClassifier().set_params(**{
        'n_estimators': 3100,
        'max_depth': 13,
        'min_samples_split': 70,
        'min_samples_leaf': 10,
        'criterion': 'entropy',
        'random_state': 0
    })

    return train_et(clf)



def predict(model):
    path = 'cache_%s_predict.csv' % os.path.basename(__file__)

    if os.path.exists(path):
        X = pd.read_csv(path, parse_dates=['Date_received'])
    else:
        offline, online = get_preprocess_data()

        # 2016-03-16 ~ 2016-06-30
        start = '2016-03-16'
        offline = offline[(offline.Coupon_id == 0) & (start <= offline.Date) | (start <= offline.Date_received)]
        online = online[(online.Coupon_id == 0) & (start <= online.Date) | (start <= online.Date_received)]

        X = get_preprocess_data(True)
        X = get_offline_features(X, offline)
        X = get_online_features(online, X)
        X.drop_duplicates(inplace=True)
        X.fillna(0, inplace=True)
        X.to_csv(path, index=False)

    sample_submission = X[['User_id', 'Coupon_id', 'Date_received']].copy()
    sample_submission.Date_received = sample_submission.Date_received.dt.strftime('%Y%m%d')
    drop_columns(X, True)

    if model is 'blending':
        predict = blending(X)
    else:
        clf = eval('train_%s' % model)()
        predict = clf.predict_proba(X)[:, 1]

    sample_submission['Probability'] = predict
    sample_submission.to_csv('submission_%s.csv' % model,
                             #  float_format='%.5f',
                             index=False, header=False)


def blending(predict_X=None):
    global log
    log += '\n'

    X = get_train_data().drop(columns='Coupon_id')
    y = X.pop('label')

    X = np.asarray(X)
    y = np.asarray(y)

    _, X_submission, _, y_test_blend = train_test_split(X, y,
                                                        random_state=0
                                                        )

    if predict_X is not None:
        X_submission = np.asarray(predict_X)

    X, _, y, _ = train_test_split(X, y,
                                  train_size=100000,
                                  random_state=0
                                  )

    # np.random.seed(0)
    # idx = np.random.permutation(y.size)
    # X = X[idx]
    # y = y[idx]

    skf = StratifiedKFold()
    clfs = ['gbdt', 'xgb', 'lgb', 'cat',
            # 'rf_gini', 'et_gini'
            ]

    blend_X_train = np.zeros((X.shape[0], len(clfs)))
    blend_X_test = np.zeros((X_submission.shape[0], len(clfs)))

    for j, v in enumerate(clfs):
        clf = eval('train_%s' % v)(True)

        aucs = []
        dataset_blend_test_j = []

        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            clf = fit_eval_metric(clf, X_train, y_train)

            y_submission = clf.predict_proba(X_test)[:, 1]
            aucs.append(roc_auc_score(y_test, y_submission))

            blend_X_train[test_index, j] = y_submission
            dataset_blend_test_j.append(clf.predict_proba(X_submission)[:, 1])

        log += '%7s' % v + ' auc: %f\n' % np.mean(aucs)
        blend_X_test[:, j] = np.asarray(dataset_blend_test_j).T.mean(1)

    print('blending')
    clf = LogisticRegression()
    # clf = GradientBoostingClassifier()
    clf.fit(blend_X_train, y)
    y_submission = clf.predict_proba(blend_X_test)[:, 1]

    # Linear stretch of predictions to [0,1]
    y_submission = (y_submission - y_submission.min()) / (y_submission.max() - y_submission.min())
    if predict_X is not None:
        return y_submission
    log += '\n  blend auc: %f\n\n' % roc_auc_score(y_test_blend, y_submission)


if __name__ == '__main__':
    start = datetime.datetime.now()
    print(start.strftime('%Y-%m-%d %H:%M:%S'))
    log = '%s\n' % start.strftime('%Y-%m-%d %H:%M:%S')
    cpu_jobs = os.cpu_count() - 1
    date_null = pd.to_datetime('1970-01-01', format='%Y-%m-%d')

    # analysis()
    # detect_duplicate_columns()
    # feature_importance_score()

    grid_search_gbdt()
    # train_gbdt()
    # predict('gbdt')

    # grid_search_xgb()
    # train_xgb()
    # predict('xgb')

    # grid_search_lgb()
    # train_lgb()
    # predict('lgb')

    # grid_search_cat()
    # train_cat()
    # predict('cat')

    # grid_search_rf()
    # train_rf_gini()
    # predict('rf_gini')

    # grid_search_rf('entropy')
    # train_rf_entropy()
    # predict('rf_entropy')

    # grid_search_et()
    # train_et_gini()
    # predict('et_gini')

    # grid_search_et('entropy')
    # train_et_entropy()
    # predict('et_entropy')

    # blending()
    # predict('blending')

    log += 'time: %s\n' % str((datetime.datetime.now() - start)).split('.')[0]
    log += '----------------------------------------------------\n'
    open('%s.log' % os.path.basename(__file__), 'a').write(log)
    print(log)




