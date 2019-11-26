# -*- coding: utf-8 -*-

"""
@author: hsowan <hsowan.me@gmail.com>
@date: 2019/11/22

Use TF-IDF + XGBoost

Best: 0.534406

"""

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import csv

# read persist object
with open('data.pkl', 'rb') as f:
    train_data = pickle.load(f)
    test_data = pickle.load(f)

# ngram_range: All values of n such that min_n <= n <= max_n will be used.
vectorizer = TfidfVectorizer(ngram_range=(1, 5), min_df=3, max_df=0.9)

# type: scipy.sparse.csr.csr_matrix
# return X : sparse matrix, [n_samples, n_features]
train_features = vectorizer.fit_transform(train_data['apis'].tolist())
test_features = vectorizer.transform(test_data['apis'].tolist())

X, y = train_features, train_data['label'].tolist()

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

d_train = xgb.DMatrix(X_train, label=y_train)
d_val = xgb.DMatrix(X_val, label=y_val)
d_test = xgb.DMatrix(test_features)

# https://xgboost.readthedocs.io/en/latest/parameter.html#general-parameters
# eta: learning_rate: {0.1: 0.534406, 0.2: 0.568074, 0.07: unknown}
# eval_metric: mlogloss multiclass logloss: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html
# max_depth: Maximum depth of a tree. Increasing this value will make the model more complex and more likely to overfit
# objective: softprob output a vector of ndata * nclass
param = {'max_depth': 6, 'eta': 0.1, 'eval_metric': 'mlogloss', 'verbosity': 0, 'objective': 'multi:softprob',
         'num_class': 8, 'subsample': 0.8,
         'colsample_bytree': 0.85}

evallist = [(d_train, 'train'), (d_val, 'val')]

num_round = 300
bst = xgb.train(param, d_train, num_round, evallist, early_stopping_rounds=50)

preds = bst.predict(d_test)
out = []
for i in range(test_data.shape[0]):
    tmp = []
    probs = preds[i].tolist()
    # file_id
    tmp.append(i + 1)
    tmp.extend(probs)
    out.append(tmp)

# param newline='' for csv_file
with open('result_xgboost.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['file_id', 'prob0', 'prob1', 'prob2', 'prob3', 'prob4', 'prob5', 'prob6', 'prob7'])
    writer.writerows(out)
