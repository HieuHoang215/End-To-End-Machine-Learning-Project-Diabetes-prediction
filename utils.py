# Importing core libraries
import numpy as np
import pandas as pd
from joblib import dump, load
import json
import random
import os

# Suppressing warnings because of skopt verbosity
import warnings
warnings.filterwarnings("ignore")
from collections import Counter
# Classifier/Regressor
from xgboost import XGBClassifier

# Data processing
from sklearn.preprocessing import OrdinalEncoder, PolynomialFeatures, MinMaxScaler, QuantileTransformer, KBinsDiscretizer, StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, FactorAnalysis, FastICA
from sklearn.mixture import GaussianMixture
from sklearn.feature_selection import SelectKBest, VarianceThreshold, mutual_info_classif
from scipy.stats import zscore
from imblearn.over_sampling import SMOTE
# Models
from xgboost import XGBClassifier

# Metrics
from sklearn.metrics import f1_score

# For reproducibility
def set_seed(seed):
    """
    Seeds basic parameters for reproducibility of results
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

# Save models function
def save_preprocessing_models(fa, mi_selector, vt_selector, transformer, kbins, encoder, save_path):
    models = {
        'fa': fa,
        'mi_selector': mi_selector,
        'vt_selector': vt_selector,
        'transformer': transformer,
        'kbins': kbins,
        'encoder': encoder
    }
    dump(models, save_path)

# Load models function
def load_preprocessing_models(load_path):
    models = load(load_path)
    return models['fa'], models['mi_selector'], models['vt_selector'], models['transformer'], models['kbins'], models['encoder']

def preprocess(df, split='train'):
    if split != 'test':
        X = df.drop('two_year_recid', axis=1)
        y = df.two_year_recid
    else:
        X = df
        y = None

    categoricals = X.columns.tolist()
    high_cardinalities = ['age', 'priors_count']

    if split == 'train':
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X[['sex', 'race', 'c_charge_degree']] = encoder.fit_transform(X[['sex', 'race', 'c_charge_degree']])
        # Factor Analysis
        fa = FactorAnalysis(n_components=len(categoricals), rotation='varimax', random_state=0)
        fa.fit(X[categoricals])

        extra_feats = [f'fa_{i}' for i in range(len(categoricals))]
        X[extra_feats] = fa.transform(X[categoricals])

        # Mutual Information
        mi_selector = SelectKBest(mutual_info_classif, k=5)
        X_mi = mi_selector.fit_transform(X, y)

        mi_feats = [f'mi_{i}' for i in range(X_mi.shape[1])]
        X[mi_feats] = X_mi

        extra_feats += mi_feats

        # Variance Threshold
        vt_selector = VarianceThreshold()
        X_vt = vt_selector.fit_transform(X, y)

        vt_feats = [f'vt_{i}' for i in range(X_vt.shape[1])]
        X[vt_feats] = X_vt

        extra_feats += vt_feats

        # Quantile Transformer
        transformer = QuantileTransformer(output_distribution='uniform', random_state=0)
        X_qt = transformer.fit_transform(X[high_cardinalities])

        qt_feats = [f'qt_{i}' for i in range(X_qt.shape[1])]
        X[qt_feats] = X_qt

        extra_feats += qt_feats

        # Discretizer
        kbins = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform', random_state=0)
        X_kbins = kbins.fit_transform(X[high_cardinalities])

        kbins_feats = [f'kbins_{i}' for i in range(X_kbins.shape[1])]
        X[kbins_feats] = X_kbins

        extra_feats += kbins_feats

        # Apply SMOTE on the training set
        print('Original dataset shape:', Counter(y))
        sm = SMOTE(random_state=42)
        X, y = sm.fit_resample(X, y)
        print('Resampled dataset shape:', Counter(y))

        # Save models for later use
        save_preprocessing_models(fa, mi_selector, vt_selector, transformer, kbins, encoder, 'preprocessing_models.joblib')

        return X, y

    else:  # split == 'validation' or split == 'test'
        # Load models
        fa, mi_selector, vt_selector, transformer, kbins, encoder = load_preprocessing_models('preprocessing_models.joblib')
        # Encode string-categorical features
        X[['sex', 'race', 'c_charge_degree']] = encoder.transform(X[['sex', 'race', 'c_charge_degree']])

        # Factor Analysis
        extra_feats = [f'fa_{i}' for i in range(len(categoricals))]
        X[extra_feats] = fa.transform(X[categoricals])

        # Mutual Information
        X_mi = mi_selector.transform(X)

        mi_feats = [f'mi_{i}' for i in range(X_mi.shape[1])]
        X[mi_feats] = X_mi
        extra_feats += mi_feats

        # Variance Threshold
        X_vt = vt_selector.transform(X)

        vt_feats = [f'vt_{i}' for i in range(X_vt.shape[1])]
        X[vt_feats] = X_vt
        extra_feats += vt_feats

        # Quantile Transformer
        X_qt = transformer.transform(X[high_cardinalities])

        qt_feats = [f'qt_{i}' for i in range(X_qt.shape[1])]
        X[qt_feats] = X_qt

        extra_feats += qt_feats

        # Discretizer
        X_kbins = kbins.transform(X[high_cardinalities])

        kbins_feats = [f'kbins_{i}' for i in range(X_kbins.shape[1])]
        X[kbins_feats] = X_kbins

        extra_feats += kbins_feats

        return X, y

def train(X_train, y_train, X_dev, y_dev, model_name):
    with open(f'best_hyperparameters_{model_name}.json') as f:
        params = json.load(f)

    model = XGBClassifier(random_state=0,
                          objective='binary:logistic',
                          n_estimators=100,
                          tree_method="exact",
                          **params)

    model.fit(X_train, y_train, early_stopping_rounds=300, eval_set=[(X_dev, y_dev)], verbose=0)

    return model
    #
    # elif model_name == 'catboost':
    #     model = CatBoostClassifier(
    #         random_state=0,
    #         verbose=0,
    #         task_type="CPU",
    #         **params,
    #     )
    #     model.fit(X[features], y)
    #
    #     return model, features
    # elif model_name == 'gb':
    #     model = GradientBoostingClassifier(random_state=0, **params)
    #     model.fit(X[features], y)
    #
    #     return model, features

def predict(model, X, y, name):
    predictions = model.predict(X)
    probas = model.predict_proba(X)
    print(f"{name}: {f1_score(y, predictions)}")
    return predictions, probas

# def meta_predict(X, model_dict):
#     meta_features = []
#     for model_name, model in model_dict.items():
#         if model_name != 'meta_model':
#             features = get_features(model_name)
#             meta_features.append(model.predict_proba(X[features]))
#     meta_model = model_dict['meta_model']
#     meta_features = np.column_stack(meta_features)
#     predictions = meta_model.predict(meta_features)
#     return predictions
#
# def get_features(model_name):
#     with open(f'selected_features_{str(model_name)[:-1]}1.txt') as f:
#         features = [line.strip() for line in f]
#     return features