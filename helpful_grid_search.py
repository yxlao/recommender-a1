from __future__ import print_function
from collections import defaultdict
import numpy as np
import scipy as sp
import cPickle as pickle
import time
import os
import datetime

# data
data_root = os.path.expanduser("~") + '/data/CSE255/'

# load all_data
print("loading data")
start_time = time.time()
all_data = pickle.load(open(data_root + "all_data.pickle", "rb"))
print(time.time() - start_time)

# split training and valid set
all_size = len(all_data)
train_size = 900000
# train_size = all_size # uncomment this to produce test
train_data = all_data[:train_size]
valid_size = 100000
valid_data = all_data[all_size - valid_size:]

# remove the outlier
print("removing outlier")
for i in reversed(range(train_size)):
    d = train_data[i]
    if d['helpful']['outOf'] > 5000:
        train_data.pop(i)

# utility functions
def get_mae(helpfuls, helpfuls_predict):
    return np.sum(np.fabs(helpfuls_predict - helpfuls.astype(float))) / helpfuls.shape[0]

# get global average
train_helpfuls = np.array([d['helpful']['nHelpful'] for d in train_data])
train_outofs = np.array([d['helpful']['outOf'] for d in train_data])
train_avg_ratio = np.sum(train_helpfuls) / np.sum(train_outofs.astype(float))
print('avg helpfulness ratio', train_avg_ratio)

# linear search best ratio
def linear_search_ratio(helpfuls, outofs, search_range=(0.3, 1.0, 0.001)):
    alphas = np.arange(*search_range)
    errors = [get_mae(helpfuls, outofs * alpha) for alpha in alphas]
    optimal_alpha = alphas[np.argmin(errors)]
    return optimal_alpha

# training set global
train_helpfuls = np.array([d['helpful']['nHelpful'] for d in train_data])
train_outofs = np.array([d['helpful']['outOf'] for d in train_data])
train_avg_ratio = linear_search_ratio(
    train_helpfuls, train_outofs, search_range=(0.3, 1.0, 0.001))
print('optimal helpfulness ratio', train_avg_ratio)

# get average for a user
users_outof = dict()
users_helpful = dict()

for d in train_data:
    user_id = d['reviewerID']
    users_outof[user_id] = users_outof.get(
        user_id, 0.0) + float(d['helpful']['outOf'])
    users_helpful[user_id] = users_helpful.get(
        user_id, 0.0) + float(d['helpful']['nHelpful'])

users_ratio = dict()
for user_id in users_outof:
    if users_outof[user_id] != 0:
        users_ratio[user_id] = users_helpful[user_id] / users_outof[user_id]
    else:
        users_ratio[user_id] = train_avg_ratio

# get average for a item
items_outof = dict()
items_helpful = dict()

for d in train_data:
    item_id = d['itemID']
    items_outof[item_id] = items_outof.get(
        item_id, 0.0) + float(d['helpful']['outOf'])
    items_helpful[item_id] = items_helpful.get(
        item_id, 0.0) + float(d['helpful']['nHelpful'])

items_ratio = dict()
for item_id in items_outof:
    if items_outof[item_id] != 0:
        items_ratio[item_id] = items_helpful[item_id] / items_outof[item_id]
    else:
        items_ratio[item_id] = train_avg_ratio

# pre-computed features
with open('betas.pickle') as f:
    beta_us, beta_is = pickle.load(f)

with open('train_ratio_list.pickle') as f:
    train_ratio_list = pickle.load(f)

with open(os.path.join(data_root, 'num_unique_word.feature')) as f:
    num_unique_word_dict = pickle.load(f)

with open(os.path.join(data_root, 'style_dict.feature')) as f:
    style_dict = pickle.load(f)
    # style_dict['U243261361']['I572782694']
    # {'avg_word_len': 4.857142857142857,
    #  'capital_count': 11.0,
    #  'capital_ratio': 0.028205128205128206,
    #  'dotdotdot_count': 4.0,
    #  'exclam_count': 0.0,
    #  'exclam_exclam_count': 0.0,
    #  'num_chars': 369.0,
    #  'num_sentences': 3.0,
    #  'num_unique_words': 50,
    #  'num_words': 63.0,
    #  'num_words_summary': 2,
    #  'punctuation_count': 21.0,
    #  'punctuation_ratio': 0.05384615384615385,
    #  'question_count': 0.0,
    #  'redability': 16.65714285714285}

def get_y_m_d(d):
    unix_time = d['unixReviewTime']
    y, m, d = datetime.datetime.fromtimestamp(
        unix_time).strftime('%Y-%m-%d').split('-')
    y = int(y)
    m = int(m)
    d = int(d)
    return(y, m, d)

def get_feature_time(d):
    y, m, d = get_y_m_d(d)
    y = min(y, 2014)
    y = max(y, 1996)
    # 1996 [1,0,..,0] 2014 [0,0,...,0]
    y_feature = [0] * (2014 - 1996 + 1)
    y_feature[y - 1996] = 1
    # jan [1,0,...,0] dec [0,0,...,0]
    m_feature = [0] * 12
    m_feature[m - 1] = 1
    # date1 [1,0,...,0] date31 [0,0,...,0]
    d_feature = [0] * 31
    d_feature[d - 1] = 1
    # concatenate
    feature = y_feature[:-1] + m_feature[:-1] + d_feature[:-1]
    return feature

def get_num_uique_word(d):
    wordCount = defaultdict(int)
    for w in d["reviewText"].split():
        w = "".join([c for c in w.lower() if not c in punctuation])
        w = stemmer.stem(w)
        wordCount[w] += 1
    return len(wordCount)

def get_feature(d):
    user_id = d['reviewerID']
    item_id = d['itemID']

    feature = [1.0]
    feature += [users_ratio[user_id], items_ratio[item_id]]
    feature += [float(d['rating'])]

    s = style_dict[user_id][item_id]
    feature += [s['num_words'],
                s['redability'],
                s['exclam_exclam_count'] + s['question_count'],
                s['capital_ratio'],
                s['dotdotdot_count'],
                s['num_unique_words']
               ]

    feature += get_feature_time(d)

    return feature

# get [feature, label] from single datum
def get_feature_and_ratio_label(d, users_ratio, items_ratio):
    # check valid
    outof = float(d['helpful']['outOf'])
    if outof == 0:
        raise('out of cannot be 0 for ratio')

    # get feature and ratio
    feature = get_feature(d)
    ratio_label = float(d['helpful']['nHelpful']) / \
        float(d['helpful']['outOf'])
    return (feature, ratio_label)

# build [feature, label] list from entire dataset
def make_average_regression_dataset(train_data, users_ratio, items_ratio):
    features = []
    labels = []

    for d in train_data:
        if float(d['helpful']['outOf']) == 0:
            continue
        feature, label = get_feature_and_ratio_label(
            d, users_ratio, items_ratio)
        features.append(feature)
        labels.append(label)
    return (np.array(features), np.array(labels))

# make one prediction
def predict_helpful(d, ratio_predictor, train_avg_ratio, users_ratio,
                    items_ratio):
    # ratio_predictor[func]: y = ratio_predictor(get_feature(d))
    user_id = d['reviewerID']
    item_id = d['itemID']
    outof = float(d['helpful']['outOf'])

    if (user_id in users_ratio) and (item_id in items_ratio):
        # ratio = np.dot(get_feature(d), theta)
        predict = ratio_predictor(np.array(get_feature(d)).reshape(1, -1))
        ratio = predict[0]  # np.ndarray
    elif (user_id in users_ratio) and (item_id not in items_ratio):
        ratio = users_ratio[user_id]
    elif (user_id not in users_ratio) and (item_id in items_ratio):
        ratio = items_ratio[item_id]
    else:
        ratio = train_avg_ratio
    return ratio * outof

# make predictions and get mae on a dataset
def get_valid_mae(valid_data, ratio_predictor, train_avg_ratio, users_ratio,
                  items_ratio):
    # ground truth nhelpful
    helpfuls = np.array([float(d['helpful']['nHelpful']) for d in valid_data])
    # predited nhelpful
    helpfuls_predict = np.array([predict_helpful(d, ratio_predictor,
                                                 train_avg_ratio, users_ratio,
                                                 items_ratio) for d in valid_data])
    # return mae
    return get_mae(helpfuls, helpfuls_predict)

###################### Gradient Boosting Grid Search ######################

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.grid_search import GridSearchCV

# build dataset
# train_xs, train_ys = make_average_regression_dataset(train_data,
#                                                      users_ratio, items_ratio)
# valid_xs, valid_ys = make_average_regression_dataset(valid_data,
#                                                      users_ratio, items_ratio)
all_xs, all_ys = make_average_regression_dataset(all_data, users_ratio,
                                                 items_ratio)
print("dataset prepared")

# set grid search param
param_grid = {'learning_rate': [0.05, 0.02, 0.01, 0.005, 0.002, 0.001],
              'max_depth': [3, 4, 6],
              'min_samples_leaf': [3, 5, 9, 17],
              'max_features': [1.0, 0.5, 0.3, 0.1]
              }

param_grid = {'learning_rate': [0.1, 0.05],
              'max_depth': [3, 4]
              }

est = GradientBoostingRegressor(n_estimators=1000, loss='lad')

gs_cv = GridSearchCV(est,
                     param_grid,
                     verbose=1,
                     n_jobs=18,
                     pre_dispatch=32)

gs_cv.fit(train_xs[:5000], train_ys[:5000])

print(gs_cv.best_params_)

# import ipdb; ipdb.set_trace()

# gbr = GradientBoostingRegressor(learning_rate=0.005,
#                                 n_estimators=1000,
#                                 max_depth=6,
#                                 max_features=0.1,
#                                 min_samples_leaf=9, loss='lad')
# gbr.fit(train_xs[:5000], train_ys[:5000])
#
# print(get_valid_mae(valid_data,
#                     gbr.predict,
#                     train_avg_ratio, users_ratio, items_ratio))
#
