from __future__ import print_function
from collections import defaultdict
import numpy as np
import scipy as sp
import cPickle as pickle
import time
from pprint import pprint
import os
import datetime
import gc

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.grid_search import GridSearchCV

from util_feature import make_dataset
import itertools


class HelpfulGridSearcher(object):

    def __init__(self, param_grid, n_estimators, n_jobs, apply_weights=False):
        super(HelpfulGridSearcher, self).__init__()
        self.param_grid = param_grid
        self.n_estimators = n_estimators
        self.n_jobs = n_jobs
        self.apply_weights = apply_weights

    def run(self):
        # load all_data and test_data
        start_time = time.time()
        all_data = pickle.load(open('all_data.pickle', 'rb'))
        print('data loaded, elapsed time:', time.time() - start_time)

        # remove the outlier
        for i in reversed(range(len(all_data))):
            d = all_data[i]
            if d['helpful']['outOf'] > 3000:
                all_data.pop(i)
            elif d['helpful']['outOf'] < d['helpful']['nHelpful']:
                all_data.pop(i)

        # build dataset
        all_xs, all_ys, all_weights = make_dataset(all_data)
        all_data = None
        gc.collect()
        print('dataset prepared, elapsed time:', time.time() - start_time)

        # init regressor
        regressor = GradientBoostingRegressor(n_estimators=self.n_estimators,
                                              loss='lad',
                                              verbose=1)

        # init grid searcher
        if self.apply_weights:
            grid_searcher = GridSearchCV(regressor,
                                         self.param_grid,
                                         fit_params={
                                             'sample_weight': all_weights},
                                         verbose=1,
                                         n_jobs=4)
        else:
            grid_searcher = GridSearchCV(regressor,
                                         self.param_grid,
                                         verbose=1,
                                         n_jobs=self.n_jobs)

        # fit grid searcher
        grid_searcher.fit(all_xs, all_ys)

        # print best params
        opt_params = grid_searcher.best_params_
        print(opt_params)

        # store optimal regressor
        opt_regressor = grid_searcher.best_estimator_

        opt_regressor_name = "opt_%s_%s_%s_%s_%s_%s" % \
                             (learning_rate,
                              max_depth,
                              min_samples_leaf,
                              max_features,
                              subsample,
                              "weight" if self.apply_weights else "no-weight")
        pickle.dump(opt_regressor, open(opt_regressor_name + '.pickle', 'wb'),
                    protocol=pickle.HIGHEST_PROTOCOL)
        opt_regressor = pickle.load(open(opt_regressor_name + '.pickle', 'rb'))

        print("saved in:", opt_regressor_name + '.pickle')

        return (opt_regressor, opt_params)


class RegressorFitDumper(object):

    def __init__(self, param_grid, n_estimators, n_jobs, apply_weights=False):
        self.param_grid = param_grid
        self.n_estimators = n_estimators
        self.n_jobs = n_jobs
        self.apply_weights = apply_weights

    def run(self):
        # # load all_data and test_data
        # print('loading data')
        start_time = time.time()
        # all_data = pickle.load(open('all_data.pickle', 'rb'))
        # print('data loaded, elapsed time:', time.time() - start_time)

        # # remove the outlier
        # for i in reversed(range(len(all_data))):
        #     d = all_data[i]
        #     if d['helpful']['outOf'] > 3000:
        #         all_data.pop(i)
        #     elif d['helpful']['outOf'] < d['helpful']['nHelpful']:
        #         all_data.pop(i)

        # # build dataset
        # all_xs, all_ys, all_weights = make_dataset(all_data)
        # pickle.dump((all_xs, all_ys, all_weights),
        #             open('zotac_all_xs_all_ys_all_weights.pickle', 'wb'),
        #             protocol=pickle.HIGHEST_PROTOCOL)
        all_xs, all_ys, all_weights = pickle.load(open('zotac_all_xs_all_ys_all_weights.pickle', 'rb'))

        # all_data = None
        # gc.collect()
        print('dataset prepared, elapsed time:', time.time() - start_time)

        for (learning_rate,
             max_depth,
             min_samples_leaf,
             max_features,
             subsample) in itertools.product(self.param_grid['learning_rate'],
                                             self.param_grid['max_depth'],
                                             self.param_grid['min_samples_leaf'],
                                             self.param_grid['max_features'],
                                             self.param_grid['subsample']):
            # print name
            opt_regressor_name = "opt_%s_%s_%s_%s_%s_%s" % \
                                 (learning_rate,
                                  max_depth,
                                  min_samples_leaf,
                                  max_features,
                                  subsample,
                                  "weight" if self.apply_weights else "no-weight")
            print(opt_regressor_name)

            # init regressor
            regressor = GradientBoostingRegressor(n_estimators=self.n_estimators,
                                                  loss='lad',
                                                  verbose=1,
                                                  learning_rate=learning_rate,
                                                  max_depth=max_depth,
                                                  min_samples_leaf=min_samples_leaf,
                                                  max_features=max_features,
                                                  subsample=subsample)
            # fit
            regressor.fit(all_xs, all_ys)

            pickle.dump(regressor, open('models_dump/' + opt_regressor_name + '.pickle', 'wb'),
                        protocol=pickle.HIGHEST_PROTOCOL)
            regressor = pickle.load(
                open('models_dump/' + opt_regressor_name + '.pickle', 'rb'))

            print("saved in:", opt_regressor_name + '.pickle')
