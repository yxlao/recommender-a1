from helpful_grid_search import HelpfulGridSearcher

# set params
param_grid = {'learning_rate': [0.2, 0.15, 0.1, 0.05],
              'max_depth': [6],
              'min_samples_leaf': [9],
              'max_features': [0.4, 0.6],
              'subsample': [0.4, 0.5, 0.7]
              }

# execute
searcher = HelpfulGridSearcher(param_grid=param_grid,
                               n_estimators=3000,
                               n_jobs=2,
                               apply_weights=False)
opt_regressor, opt_params = searcher.run()
