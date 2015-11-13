from helpful_grid_search import HelpfulGridSearcher

# set params
n_estimators = 3000
n_jobs = 4

# [zotac-torun: study learning_rate vs. subsample]
param_grid = {'learning_rate': [0.1, 0.05, 0.02, 0.01],
              'max_depth': [4],
              'min_samples_leaf': [9],
              'max_features': [0.5],
              'subsample': [0.1, 0.3, 0.5]
              }

# execute
searcher = HelpfulGridSearcher(param_grid, n_estimators, n_jobs)
opt_regressor, opt_params = searcher.run()
