# read the dumped feature and provide funcs for getting features
from __future__ import print_function
from collections import defaultdict
import numpy as np
import scipy as sp
import cPickle as pickle
import time
from pprint import pprint
import os
import datetime

# user, item feature
global_feature, users_feature, items_feature = pickle.load(
    open('global_users_items_feature.feature', 'rb'))

# style_dict
style_dict = pickle.load(open('style_dict.feature', 'rb'))


def get_feature_cat(d):
    def get_level_zero_feature(d):
        cat_list_list = d['category']
        for cat_list in cat_list_list:
            if (cat_list[0] == 'Kindle Store'):
                return [1.0]
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

    feature = []
    # how many cats
    feature += [float(len(d['category']))]
    # 2 level features
    feature += get_level_zero_feature(d)
    feature += get_level_one_feature(d)
    return feature


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
    """ return the array index ratio to insert spot """
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
               # s['avg_rating'], # comment this if using old models
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
               # s['avg_rating'], # comment this if using old models
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


def get_feature_label_weight(d, total_outof_weights):
    """ get [feature, label] from single datum """

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


def make_dataset(train_data):
    """ build [feature, label] list from entire dataset """

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


def get_mae(helpfuls, helpfuls_predict):
    """ return mean squared error """
    return np.mean(np.fabs(helpfuls_predict - helpfuls.astype(float)))


def predict_helpful(d, ratio_predictor):
    """ make one prediction """

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


def get_valid_mae(valid_data, ratio_predictor):
    """make predictions and get mae on a dataset"""

    # ground truth nhelpful
    helpfuls = np.array([float(d['helpful']['nHelpful']) for d in valid_data])
    # predited nhelpful
    helpfuls_predict = np.array(
        [predict_helpful(d, ratio_predictor) for d in valid_data])
    # return mae
    return get_mae(helpfuls, helpfuls_predict)
