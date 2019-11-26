# -*- coding: utf-8 -*-

"""
@author: hsowan <hsowan.me@gmail.com>
@date: 2019/11/23

Train & Predict

"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold

feature_path = './feature/'
result_path = './result/'

xgb_params = {'objective': 'multi:softprob',
              'num_class': 8,
              'eta': 0.04,
              'max_depth': 6,
              'subsample': 0.9,
              'colsample_bytree': 0.7,
              'lambda': 2,
              'alpha': 2,
              'gamma': 1,
              'scale_pos_weight': 20,
              'eval_metric': 'mlogloss',
              'silent': 0,
              'seed': 149}

# load feature v1
train_1 = pd.read_csv(feature_path + 'train_base_features_v1.csv')
test_1 = pd.read_csv(feature_path + 'test_base_features_v1.csv')

# load feature v2
train_2 = pd.read_csv(feature_path + 'train_base_features_v2.csv')
test_2 = pd.read_csv(feature_path + 'test_base_features_v2.csv')

interaction_feat = train_2.columns[train_2.columns.isin(test_2.columns.values)].values
train_2 = train_2[interaction_feat]
test_2 = test_2[interaction_feat]

# merge all features
train = train_1.merge(train_2, on=['file_id'], how='left')
test = test_1.merge(test_2, on=['file_id'], how='left')

# train data prepare
X = train.drop(['file_id', 'label'], axis=1)
y = train['label']

# add one_vs_rest prob
extra_feat_val = pd.read_csv(feature_path + 'tr_lr_oof_prob.csv')
extra_feat_test = pd.read_csv(feature_path + 'te_lr_oof_prob.csv')
prob_list = ['prob' + str(i) for i in range(1)]
X_extra = pd.concat([X, extra_feat_val[prob_list]], axis=1)
X_extra_test = pd.concat([test, extra_feat_test[prob_list]], axis=1)

# multi-class model training
logloss_result = []
pred_val_all = pd.DataFrame()
# 8 catagories
df_pred_test_all = pd.DataFrame(np.zeros((test.shape[0], 8)))
skf = StratifiedKFold(n_splits=5, random_state=4, shuffle=True)
# start 5-fold CV
for fold_i, (tr_index, val_index) in enumerate(skf.split(X, y)):
    print('FOLD -', fold_i, ' Start...')
    # Prepare train, val dataset
    X_train, X_val = X_extra.iloc[tr_index, :], X_extra.iloc[val_index, :]
    y_train, y_val = y[tr_index], y[val_index]

    # Train model
    # multi-class model
    d_train = xgb.DMatrix(X_train, y_train)
    d_val = xgb.DMatrix(X_val, y_val)
    d_test = xgb.DMatrix(X_extra_test.drop(['file_id'], axis=1))
    evallist = [(d_train, 'train'), (d_val, 'val')]
    num_round = 1000
    # verbose_eval=100: an evaluation metric is printed every 100 boosting stages, instead of every boosting stage
    model = xgb.train(xgb_params, d_train, num_round, evals=evallist, early_stopping_rounds=100, verbose_eval=100)
    df_pred_val = pd.DataFrame(model.predict(d_val, ntree_limit=model.best_iteration), index=X_val.index)
    df_pred_test = pd.DataFrame(model.predict(d_test, ntree_limit=model.best_iteration), index=X_extra_test.index)

    # Evaluate Model and Concatenate Val-Prediction
    m_log_loss = log_loss(y_val, df_pred_val)
    print('----------------log_loss : ', m_log_loss, ' ---------------------')
    logloss_result = logloss_result + [m_log_loss]
    truth_prob_df = pd.concat([y_val, df_pred_val], axis=1)
    pred_val_all = pd.concat([pred_val_all, truth_prob_df], axis=0)
    # Predict Test Dataset
    df_pred_test_all = df_pred_test_all + 0.2 * df_pred_test

# generate submit file
result = pd.concat([test['file_id'], df_pred_test_all], axis=1)
prob_list = ['prob' + str(i) for i in range(8)]
result.columns = ['file_id'] + prob_list
result.to_csv(result_path + 'submit.csv', index=None)


