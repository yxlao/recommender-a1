from __future__ import print_function
from collections import defaultdict
import numpy as np
import scipy as sp
import cPickle as pickle
import time
from pprint import pprint
import os
import datetime

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.grid_search import GridSearchCV


# load all_data and test_data
start_time = time.time()
all_data = pickle.load(open('all_data.pickle', 'rb'))
test_data = pickle.load(open("helpful_data.pickle", "rb"))
print('data loading time:', time.time() - start_time)

# remove the outlier
for i in reversed(range(len(all_data))):
    d = all_data[i]
    if d['helpful']['outOf'] > 3000:
        all_data.pop(i)
    elif d['helpful']['outOf'] < d['helpful']['nHelpful']:
        all_data.pop(i)

# utility functions
def get_mae(helpfuls, helpfuls_predict):
    return np.mean(np.fabs(helpfuls_predict - helpfuls.astype(float)))

# load pre computed features
global_feature, users_feature, items_feature = pickle.load(
    open('global_users_items_feature.feature', 'rb'))
style_dict = pickle.load(open('style_dict.feature', 'rb'))

###### begin cats ######
# add global_feature['level_cats'] to global_feature
raw_cats = [d['category'] for d in all_data + test_data]

level_cats = list()
for level in range(7):
    level_cats.append(set())
    for cat_list_list in raw_cats:
        for cat_list in cat_list_list:
            if len(cat_list) > level:
                level_cats[level].add(cat_list[level])
# convert set to list
for i in range(len(level_cats)):
    level_cats[i] = sorted(list(level_cats[i]))
global_feature['level_cats'] = level_cats

def get_level_zero_feature(d):
    def is_kindle(d):
        is_kindle = False
        cat_list_list = d['category']
        for cat_list in cat_list_list:
            if (cat_list[0] == 'Kindle Store'):
                return True
        return False
    if is_kindle(d):
        return [1.0]
    else:
        return [0.0]

def get_level_one_feature(d):
    cat_list_list = d['category']
    level_one_cats = set()
    for cat_list in cat_list_list:
        if len(cat_list) > 1:
            level_one_cats.add(cat_list[1])
    # gen binary feature of length 33
    feature = [0.] * len(global_feature['level_cats'][1])
    for cat in level_one_cats:
        index = global_feature['level_cats'][1].index(cat)
        feature[index] = 1.0
    return feature

def get_feature_cat(d):
    feature = []
    # how many cats
    feature += [float(len(d['category']))]
    # 2 level feature
    feature += get_level_zero_feature(d)
    feature += get_level_one_feature(d)
    return feature

###### end cats ######

# feature engineering
def get_feature_time(d):
    unix_time = d['unixReviewTime']
    y, m, d = datetime.datetime.fromtimestamp(
        unix_time).strftime('%Y-%m-%d').split('-')
    y = float(y)
    m = float(m)
    d = float(d)
    return [y, m, d]

def get_feature_style(d):
    # load from style dict
    user_id = d['reviewerID']
    item_id = d['itemID']
    s = style_dict[user_id][item_id]

    feature = [s['num_words'],
               s['num_words_summary'],
               s['redability'],
               s['avg_word_len'],
               s['num_words'] /
               s['num_sentences'] if s['num_sentences'] != 0.0 else 0.0,
               s['num_unique_words'],
               s['exclam_exclam_count'] + s['question_count'],
               s['dotdotdot_count'],
               s['capital_ratio']
               ]
    return feature

def get_time_spot_ratio(times, spot):
    # return the array index ratio to insert spot
    if len(times) == 0:
        return 0.
    index = np.searchsorted(np.array(times), spot)
    return float(index) / float(len(times))

def get_feature_user(d):
    user_id = d['reviewerID']
    unix_time = d['unixReviewTime']

    s = users_feature[user_id]
    feature = [s['ratio_a'],
               s['ratio_b'],
               s['num_reviews'],
               s['avg_review_length'],
               s['avg_summary_length'],
               get_time_spot_ratio(s['review_times'], unix_time)
               ]
    return feature

def get_feature_item(d):
    item_id = d['itemID']
    unix_time = d['unixReviewTime']

    s = items_feature[item_id]
    feature = [s['ratio_a'],
               s['ratio_b'],
               s['num_reviews'],
               s['avg_review_length'],
               s['avg_summary_length'],
               get_time_spot_ratio(s['review_times'], unix_time)
               ]
    return feature

def get_feature(d):
    user_id = d['reviewerID']
    item_id = d['itemID']
    unix_time = d['unixReviewTime']

    # offset
    feature = [1.0]

    # user
    feature += get_feature_user(d)
    # item
    feature += get_feature_item(d)

    # outof
    feature += [float(d['helpful']['outOf'])]
    # rating
    feature += [float(d['rating'])]
    # styles
    feature += get_feature_style(d)
    # time
    feature += get_feature_time(d)
    # category
    feature += get_feature_cat(d)

    for i in range(len(feature)):
        feature[i] = float(feature[i])

    return feature

# get [feature, label] from single datum
def get_feature_label_weight(d, total_outof_weights):
    # check valid
    outof = float(d['helpful']['outOf'])
    assert outof != 0.

    # feature
    feature = get_feature(d)
    # label
    ratio_label = float(d['helpful']['nHelpful']) / \
        float(d['helpful']['outOf'])
    # weight
    weight = float(d['helpful']['outOf']) / total_outof_weights

    return (feature, ratio_label, weight)

# build [feature, label] list from entire dataset
def make_dataset(train_data):
    features = []
    labels = []
    weights = []

    train_outofs = np.array([d['helpful']['outOf']
                             for d in train_data]).astype(float)
    total_outof_weights = np.sum(train_outofs)

    for d in train_data:
        if float(d['helpful']['outOf']) == 0:
            continue
        feature, label, weight = get_feature_label_weight(
            d, total_outof_weights)
        features.append(feature)
        labels.append(label)
        weights.append(weight)

    return (np.array(features), np.array(labels), np.array(weights))

# make one prediction
def predict_helpful(d, ratio_predictor):
    # ratio_predictor[func]: y = ratio_predictor(get_feature(d))

    user_id = d['reviewerID']
    item_id = d['itemID']
    outof = float(d['helpful']['outOf'])

    if (user_id in users_feature) and (item_id in items_feature):
        predict = ratio_predictor(np.array(get_feature(d)).reshape((1, -1)))
        ratio = predict[0]  # np.ndarray
    elif (user_id in users_feature) and (item_id not in items_feature):
        ratio = users_feature[user_id]['ratio_b']
    elif (user_id not in users_feature) and (item_id in items_feature):
        ratio = items_ratio[item_id]['ratio_b']
    else:
        ratio = global_feature['global_ratio_b']
    return ratio * outof

# make predictions and get mae on a dataset
def get_valid_mae(valid_data, ratio_predictor):
    # ground truth nhelpful
    helpfuls = np.array([float(d['helpful']['nHelpful']) for d in valid_data])
    # predited nhelpful
    helpfuls_predict = np.array(
        [predict_helpful(d, ratio_predictor) for d in valid_data])
    # return mae
    return get_mae(helpfuls, helpfuls_predict)

##########  Grid Search ##########

MAX_ITER = 3000

# build dataset
all_xs, all_ys, all_weights = make_dataset(all_data)
pickle.dump((all_xs, all_ys, all_weights),
            open("all_xs_all_ys_all_weights.pickle", "wb"),
            protocol=pickle.HIGHEST_PROTOCOL)
print('dataset prepared')

# set grid search param
param_grid = {'learning_rate': [0.05, 0.02, 0.01, 0.005],
              'max_depth': [6],
              'min_samples_leaf': [9],
              'max_features': [0.5],
              'subsample': [0.1]
              }

# init regressor
regressor = GradientBoostingRegressor(n_estimators=MAX_ITER,
                                      loss='lad',
                                      verbose=1)

# grid search
grid_searcher = GridSearchCV(regressor, param_grid,
                             fit_params={'sample_weight': all_weights},
                             verbose=1, n_jobs=2)
grid_searcher.fit(all_xs, all_ys)

# print best params
opt_params = grid_searcher.best_params_
print(opt_params)

# store optimal regressor
opt_regressor = grid_searcher.best_estimator_

opt_regressor_name = "opt_%s_%s_%s_%s_%s" % (opt_params['learning_rate'],
                                             opt_params['max_depth'],
                                             opt_params['min_samples_leaf'],
                                             opt_params['max_features'],
                                             opt_params['subsample'])
pickle.dump(opt_regressor, open(opt_regressor_name + '.pickle', 'wb'),
            protocol=pickle.HIGHEST_PROTOCOL)
opt_regressor = pickle.load(open(opt_regressor_name + '.pickle', 'rb'))


########## Produce Test ##########

# load helpful_data.json
test_data = pickle.load(open('helpful_data.pickle', 'rb'))

# on test set
test_helpfuls_predict = [
    predict_helpful(d, opt_regressor.predict) for d in test_data]

# load 'pairs_Helpful.txt'
# get header_str and user_item_outofs
with open('pairs_Helpful.txt') as f:
    # read and strip lines
    lines = [l.strip() for l in f.readlines()]
    # stirip out the headers
    header_str = lines.pop(0)
    # get a list of user_item_ids
    user_item_outofs = [l.split('-') for l in lines]
    user_item_outofs = [[d[0], d[1], float(d[2])] for d in user_item_outofs]

# make sure `data.json` and `pairs_Helpful.txt` the same order
for (user_id, item_id, outof), d in zip(user_item_outofs, test_data):
    assert d['reviewerID'] == user_id
    assert d['itemID'] == item_id
    assert d['helpful']['outOf'] == outof

# write to output file
f = open('predictions_Helpful.txt', 'w')
print(header_str, file=f)
for (user_id, item_id, outof), helpful_predict in zip(user_item_outofs,
                                                      test_helpfuls_predict):
    print('%s-%s-%s,%s' %
          (user_id, item_id, int(outof), round(helpful_predict)), file=f)
f.close()


print('total elapsed time:', time.time() - start_time)
