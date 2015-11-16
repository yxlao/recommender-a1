from helpful_grid_search import RegressorFitDumper

# set params
param_grid = {'learning_rate': [0.05],
              'max_depth': [4],
              'min_samples_leaf': [3, 7, 9, 13],
              'max_features': [0.3, 0.5, 0.7],
              'subsample': [0.1, 0.4, 0.5]
              }
dumper = RegressorFitDumper(param_grid=param_grid,
                            n_estimators=3000,
                            n_jobs=None,
                            apply_weights=False)
dumper.run()