import optuna
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.ensemble import VotingClassifier
from optuna.integration import XGBoostPruningCallback
from sklearn.metrics import log_loss, f1_score
from utils import *
import pandas as pd

import warnings
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective(trial):
    # search space for xgboost
    xgb_param = {
        'n_estimators': 100,
        "verbosity": 0,
        "objective": "binary:logistic",
        # use exact for small dataset.
        "tree_method": "exact",
        # defines booster, gblinear for linear functions.
        "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
        # L2 regularization weight.
        "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
        # L1 regularization weight.
        "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
        # sampling ratio for training data.
        "subsample": trial.suggest_float("subsample", 0.2, 1.0),
        # sampling according to each tree.
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
    }
    catboost_param = {
        'iterations': 100,
        'depth': trial.suggest_int('depth', 3, 5,8),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10, log=True),
        'subsample': trial.suggest_float('subsample', 0.1, 1.0),
        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.1, 1.0),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1, 10),
    }
    lgbm_param = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': trial.suggest_int('num_leaves', 10, 200),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.1, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.1, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
        'min_child_samples': trial.suggest_int('min_child_samples', 1, 20),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10, log=True),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1, 10),
        'verbosity': -1,  # Set to 0 or -1 for silent mode, 1 for some logging
        'categorical_feature': [0, 1, 3],  # Specify categorical features indices
    }
    voting_param = {
        'voting': 'soft',  # Choose between 'hard' or 'soft' voting
        'weights': (1, 1, 1),  # Adjust the weights based on model performance
    }
    if xgb_param["booster"] in ["gbtree", "dart"]:
        # maximum depth of the tree, signifies complexity of the tree.
        xgb_param["max_depth"] = trial.suggest_int("max_depth", 3, 9, step=2)
        # minimum child weight, larger the term more conservative the tree.
        xgb_param["min_child_weight"] = trial.suggest_int("min_child_weight", 2, 10)
        xgb_param["eta"] = trial.suggest_float("eta", 1e-8, 1.0, log=True)
        # defines how selective algorithm is.
        xgb_param["gamma"] = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
        xgb_param["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])

    if xgb_param["booster"] == "dart":
        xgb_param["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
        xgb_param["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
        xgb_param["rate_drop"] = trial.suggest_float("rate_drop", 1e-8, 1.0, log=True)
        xgb_param["skip_drop"] = trial.suggest_float("skip_drop", 1e-8, 1.0, log=True)

    model1 = XGBClassifier(
        random_state=0,
        **xgb_param,
    )
    model2 = CatBoostClassifier(
        random_state=0,
        **catboost_param,
    )
    model3 = LGBMClassifier(
        random_state=0,
        **lgbm_param,
    )
    voting_model = VotingClassifier(
        estimators=[('xgb', model1), ('cat', model2), ('lgbm', model3)],
        **voting_param,
    )

    X_train, y_train, X_dev, y_dev = load_data()
    # Fit the XGBoost model
    model1.fit(X_train, y_train, early_stopping_rounds=500, eval_set=[(X_dev, y_dev)], verbose=0,
              callbacks=[XGBoostPruningCallback(trial, 'validation_0-logloss')])
    # Fit the CatBoost model
    model2.fit(X_train, y_train, eval_set=[(X_dev, y_dev)], early_stopping_rounds=500, verbose=0)

    # Fit the LightGBM model
    early_stopping_cb = lgb.early_stopping(stopping_rounds=500, first_metric_only=True, verbose=True)

    model3.fit(X_train, y_train,  eval_set=[(X_dev, y_dev)],callbacks=[early_stopping_cb])

    # Fit the VotingClassifier to the training data
    voting_model.fit(X_train, y_train)
    preds = voting_model.predict(X_dev)
    f1 = f1_score(y_dev, preds)
    return f1

def create_study(name):
    global model_name
    model_name = name
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)

    # Specify the file path where you want to save the hyperparameters
    file_path = f'best_hyperparameters_{model_name}.json'

    # Save the best hyperparameters to a JSON file
    with open(file_path, 'w') as file:
        json.dump(study.best_params, file)
    print(study.best_params)
    print(study.best_value)
    print(f"Best hyperparameters saved to {file_path}")

def load_data():
    X_train = pd.read_csv('preprocessed_X_train.csv')
    y_train = pd.read_csv('y_train.csv')
    X_dev = pd.read_csv('preprocessed_X_dev.csv')
    y_dev = pd.read_csv('y_dev.csv')

    # with open(f'selected_features_{str(model_name)[:-1]}1.txt') as f:
    #     features = [line.strip() for line in f]

    return X_train, y_train, X_dev, y_dev
