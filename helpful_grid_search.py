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


class HelpfulGridSearcher(object):

    def __init__(self, param_grid, n_estimators, n_jobs):
        super(HelpfulGridSearcher, self).__init__()
        self.param_grid = param_grid
        self.n_estimators = n_estimators
        self.n_jobs = n_jobs

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

        # grid search
        grid_searcher = GridSearchCV(regressor, self.param_grid, verbose=1,
                                     n_jobs=self.n_jobs)
        grid_searcher.fit(all_xs, all_ys)

        # print best params
        opt_params = grid_searcher.best_params_
        print(opt_params)

        # store optimal regressor
        opt_regressor = grid_searcher.best_estimator_

        opt_regressor_name = "opt_%s_%s_%s_%s_%s" % (opt_params['learning_rate'],
                                                     opt_params['max_depth'],
                                                     opt_params[
                                                         'min_samples_leaf'],
                                                     opt_params[
                                                         'max_features'],
                                                     opt_params['subsample'])
        pickle.dump(opt_regressor, open(opt_regressor_name + '.pickle', 'wb'),
                    protocol=pickle.HIGHEST_PROTOCOL)
        opt_regressor = pickle.load(open(opt_regressor_name + '.pickle', 'rb'))

        return (opt_regressor, opt_params)
