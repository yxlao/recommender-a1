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

# load all_data and test_data
start_time = time.time()
all_data = pickle.load(open('all_data.pickle', 'rb'))
# remove the outlier
for i in reversed(range(len(all_data))):
    d = all_data[i]
    if d['helpful']['outOf'] > 3000:
        all_data.pop(i)
    elif d['helpful']['outOf'] < d['helpful']['nHelpful']:
        all_data.pop(i)
print('data loaded, elapsed time:', time.time() - start_time)

# build dataset
all_xs, all_ys, all_weights = make_dataset(all_data)
all_data = None
gc.collect()
print('dataset prepared, elapsed time:', time.time() - start_time)

# set grid search param
param_grid = {'learning_rate': [0.1, 0.05, 0.01, 0.005],
              'max_depth': [4, 6],
              'min_samples_leaf': [3, 9, 15],
              'max_features': [0.1, 0.5, 0.9],
              'subsample': [0.1, 0.3]
              }

# init regressor
regressor = GradientBoostingRegressor(n_estimators=3000,
                                      loss='lad',
                                      verbose=1)

# grid search
grid_searcher = GridSearchCV(regressor, param_grid, verbose=1, n_jobs=4)
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
