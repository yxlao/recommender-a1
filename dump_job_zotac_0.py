from helpful_grid_search import RegressorFitDumper

# set params
param_grid = {'learning_rate': [0.1],
              'max_depth': [5],
              'min_samples_leaf': [9],
              'max_features': [0.2, 0.5],
              'subsample': [0.4, 0.5, 0.6]
              }#running_zotac
dumper = RegressorFitDumper(param_grid=param_grid,
                            n_estimators=3000,
                            n_jobs=None,
                            apply_weights=False)
dumper.run()