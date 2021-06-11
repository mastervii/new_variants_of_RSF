from itertools import product

import pandas as pd
import numpy as np
from pathlib import Path
# import matplotlib.pyplot as plt
import sksurv
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, GridSearchCV
from sksurv.datasets import load_veterans_lung_cancer
from sksurv.datasets import get_x_y
import arff
# from sklearn.ensemble import RandomForestRegressor
from sksurv.ensemble import RandomSurvivalForest
from sksurv.nonparametric import ipc_weights

# For a full experiment
job = "ALL"
diseases = ['Angina', 'HeartAtt', 'Diabetes','Stroke','Arthritis','Alzheimer', 'Cancer', 'Psychiatric']
N_TREES = 500
INNER_FOLDS = 5
N_FOLDS = 10
D = {'max_features': [13, 10, 7, 4], 'min_samples_leaf': [5, 7, 10]}
D = {'max_features': [8, 6, 4, 3], 'min_samples_leaf': [5, 7, 10]}
test = ""

# TODO: d0 = 1,2,3

# job = "1"
# diseases = ['Angina', 'HeartAtt']
# job = "2"
# diseases = ['Diabetes','Stroke']
# job = "3"
# diseases = ['Arthritis','Alzheimer']
# job = "4"
# diseases = ['Cancer', 'Psychiatric']

# For a short run
job = "test"
diseases = ['Any-disease']
N_TREES = 5
INNER_FOLDS = 3
N_FOLDS = 2
D = {'max_features': [3], 'min_samples_leaf': [1]}
test = "_test"

def show_data(dataset):
    print(dataset.shape)
    print(dataset.describe())
    print(dataset.head())

def load_dataset(disease):
    # ELSA csv dataset
    dataset = pd.read_csv('dataset/survival-' + disease + '.csv')
    # dataset = pd.read_csv('dataset/eSHARE-' + disease + '.csv')
    if job == "test":
        dataset = dataset.sample(frac=0.1, replace=True, random_state=709)
    dataset.sort_values('target', inplace=True)
    # show_data(dataset)
    data_x, data_y = get_x_y(dataset, attr_labels=['uncensored', 'target'], pos_label=1.0)
    data_x = convert_feature_types(data_x)
    data_y['target'] = np.rint(data_y['target'])
    return data_x, data_y

def convert_feature_types(df):
    keep_numeric = ['upper_bound']
    df = df.copy()
    for col in df:
        feature = df[col]
        uniques = feature.dropna().sort_values(ascending=True, kind="quicksort").unique()
        num = len(uniques)
        if num == 2:
            one = uniques[0]
            df[col] = (feature == one).values
        elif num < 20:
            cat_dtype = pd.api.types.CategoricalDtype(categories=uniques, ordered=True)
            df[col] = feature.astype(cat_dtype)
        else:
            if 'age' not in col and col not in keep_numeric:
                df[col] = pd.cut(feature, bins=5, labels=range(5))
    return df

def imputing(X_train, X_test):
    X_train, X_test = X_train.copy(), X_test.copy()
    col_names = list(X_train)
    col_types = dict(zip(col_names, X_train.dtypes))
    # row_index = X_train.index
    # print(col_types)
    imputer = SimpleImputer(strategy='most_frequent')
    X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=col_names)
    X_train = X_train.astype(col_types)
    X_test = pd.DataFrame(imputer.transform(X_test), columns=col_names)
    X_test = X_test.astype(col_types)
    return X_train, X_test

def arff_version():
    filename = 'dataset/survival-Cancer.arff'
    data_x, data_y, _, _ = sksurv.datasets.load_arff_files_standardized(path_training=filename, attr_labels=['uncensored', 'target'], pos_label='1.0', standardize_numeric=False, to_numeric=False)
    # data_x, data_y, _, _ = sksurv.datasets.load_arff_files_standardized(path_training=filename, attr_labels=['uncensored', 'target'], pos_label='1.0')
    X, y = data_x, pd.DataFrame(data_y[['target','uncensored']])
    temp = X['upper_bound']
    X.drop('upper_bound', axis=1, inplace=True)
    X['upper_bound'] = temp
    # X, y = load_dataset('Cancer')
    # filter = X.select_dtypes(include=['category']).columns
    # X = X[filter]
    # print(X.dtypes)
    return X, y

def csv_version():
    X, y = load_dataset('Cancer')
    return X, y


def run(estimator, X_train, X_test, y_train, y_test, weights=None):
    # estimator.fit(X_train, y_train['target'].values, sample_weight=weights)
    estimator.fit(X_train, y_train)
    # y_pred = estimator.predict(X_test)
    c_val = estimator.score(X_test, y_test)
    return c_val

def inner_cv(X, y, weighted=False):
    weights = None
    i_maxes = []
    cv = StratifiedKFold(n_splits=INNER_FOLDS)
    k = 1
    for train_index, test_index in cv.split(X, y['uncensored']):
        print("inner - fold .", k)
        X_train, X_test, y_train, y_test = X.iloc[train_index], X.iloc[test_index], y[train_index], y[test_index]
        # check if column exists
        if 'upper_bound' in X_test:
            X_test = X_test.drop('upper_bound', axis=1)


        scores = []
        for i in range(len(ALL_SETTING)):
            # model = RandomForestRegressor(**ALL_SETTING[i], random_state=0, n_estimators=N_TREES, n_jobs=-1)
            model = RandomSurvivalForest(**ALL_SETTING[i], random_state=0, n_estimators=N_TREES, n_jobs=-1)
            c_val = run(model, X_train, X_test, y_train, y_test, weights=weights)
            scores.append(c_val)

        max_score = max(scores)
        i_candidate = scores.index(max_score)
        i_maxes.append(i_candidate)
        k += 1
    # get the most winning index
    i_max = max(i_maxes, key=i_maxes.count)
    best_param = ALL_SETTING[i_max]
    # best_model = RandomForestRegressor(**best_param, random_state=0, n_estimators=N_TREES, n_jobs=-1)
    best_model = RandomSurvivalForest(**best_param, random_state=0, n_estimators=N_TREES, n_jobs=-1)
    # ALL_SETTING[i]                    -> {'max_features': 13, 'min_samples_leaf': 5}
    # ALL_SETTING[i].values()           -> dict_values([13, 5])
    # tuple(ALL_SETTING[i].values())    -> (13, 5)
    candidates = [tuple(ALL_SETTING[i].values()) for i in sorted(i_maxes)]
    print(candidates)
    return best_model, tuple(best_param.values()), candidates

def inner_cv_built_in():
    model = RandomSurvivalForest( random_state=0, n_estimators=N_TREES, n_jobs=-1)
    return GridSearchCV(estimator=model, param_grid=D, cv=INNER_FOLDS, n_jobs=-1)

def rsf_test(X_train, X_test, y_train, y_test):
    X_train.drop('upper_bound', axis=1, inplace=True)
    X_test.drop('upper_bound', axis=1, inplace=True)

    model, best_param, all_params = inner_cv(X_train, y_train)
    c_val = run(model, X_train, X_test, y_train, y_test)

    # clf = inner_cv_built_in()
    # clf.fit(X_train, y_train)
    # best_param = clf.best_params_
    # c_val = clf.best_estimator_.score(X_test, y_test)
    # all_params = []

    print("c-index = ", round(c_val, 3), "(mtry,node_size) = ", best_param)
    return c_val, best_param, all_params

def temp_test(X,y):
    X.drop('upper_bound', axis=1, inplace=True)
    X, X = imputing(X, X)
    model = RandomSurvivalForest( random_state=0, n_estimators=N_TREES, n_jobs=-1)
    model.fit(X,y)
    c_val = model.score(X, y)
    print("c-index = ", round(c_val, 3))

method = "RSF_leaf_mean" + test
dir_info = './results/' + method + '/'
Path(dir_info).mkdir(parents=True, exist_ok=True)
report_columns = ['Fold', 'C-index', 'Best Args(mtry,node_size)', 'Arg Candidates']
c_vals = []
best_params = []
n_uncensoreds = []

cv = StratifiedKFold(n_splits=N_FOLDS)
ALL_SETTING = [dict(zip(D.keys(), a)) for a in product(*D.values())]
# 2017 - 2004 (wave 8 - wave 2) * 12
# TIMELINE = range(1, 7, 1)
print("Running " + method)
print("PATH: " + dir_info)

for disease in diseases:
    print("processing ", disease)
    report = []
    X, y = load_dataset(disease)
    # print(y)
    k = 1
    c_sum = 0
    selected_params = []
    # temp_test(X,y)
    # abc
    for train_index, test_index in cv.split(X, y['uncensored']):
        print(disease, "=> OUTER - fold ",k)
        X_train, X_test, y_train, y_test = X.iloc[train_index], X.iloc[test_index], y[train_index], y[test_index]
        X_train, X_test = imputing(X_train, X_test)
        if ('RSF' in method):
            c_val, param, all_params = rsf_test(X_train, X_test, y_train, y_test)
        c_sum += c_val
        selected_params.append(param)
        row = [k, round(c_val,4), param, all_params]
        # print(row)
        report.append(row)

        k += 1
    c_vals.append(round(c_sum/N_FOLDS,4))
    best_params.append(max(selected_params, key=selected_params.count))
    n_uncensoreds.append(y["uncensored"].sum())
    pd.DataFrame(report, columns=report_columns).to_csv(dir_info + method + "_" + disease + '_' + str(N_FOLDS) + '_folds.csv', index=False)
pd.DataFrame({"Disease": diseases, method: c_vals, "Params(mtry,node_size)": best_params, "Uncensored instances": n_uncensoreds}).to_csv(dir_info + method + '_Results-' + job + '.csv', index=False)