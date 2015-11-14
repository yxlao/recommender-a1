from helpful_grid_search import HelpfulGridSearcher

# set params
param_grid = {'learning_rate': [0.1, 0.05, 0.02, 0.01],
              'max_depth': [4],
              'min_samples_leaf': [3, 7, 9, 13],
              'max_features': [0.5],
              'subsample': [0.1]
              }

# execute
searcher = HelpfulGridSearcher(param_grid=param_grid,
                               n_estimators=3000,
                               n_jobs=4,
                               apply_weights=False)
opt_regressor, opt_params = searcher.run()
