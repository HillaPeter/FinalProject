import pandas as pd
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn import linear_model
import argparse

from xgboost import XGBClassifier
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler
import xgboost as xgb
import pandas as pd


from sklearn.metrics import *
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from imblearn.pipeline import Pipeline

import warnings
warnings.filterwarnings("ignore")

df_all = pd.read_csv("/home/yaararum/Files/Reop_data_for_model.csv")
path ="/home/yaararum/Files/model_outputs/"



def grid_search_Reop_UnderSampling(target,title,WithExperience=False):
    print (title)
    print("--------------------------------------------------------------------")
    print ()
    if WithExperience is False:
        X = df_all.drop(
            ['SiteID', 'surgid', 'Complics', 'STSRCHOSPD', 'STSRCOM','Reoperation'], axis=1)
        y = df_all[target]
    else:
        X = df_all.drop(
            ['SiteID', 'surgid', 'Complics', 'STSRCHOSPD', 'STSRCOM','Reoperation',
             'HospID_Reop_CABG', 'HospID_total_CABG', 'surgyear', 'HospID_total_cardiac_surgery',
             'surgid_total_cardiac_surgery', 'surgid_total_CABG', 'surgid_Reop_CABG'], axis=1)
        y = df_all[target]

    counter = Counter(y)
    # estimate scale_pos_weight value
    estimate = counter[0] / counter[1]
    print('Estimate: %.3f' % estimate)
    print(counter[0])
    print(counter[1])

    model = XGBClassifier(objective='binary:logistic', eval_metric='logloss')
    random = RandomUnderSampler(sampling_strategy='majority')
    max_depths = [1,2, 3, 5, 8, 10, 14, 18]
    learning_rates = [0.1, 0.05, 0.01]
    n_estimator = [50,100, 150,200, 300,400]
    param_grid = {'xgbclassifier__max_depth': max_depths,
                  'xgbclassifier__learning_rate': learning_rates,
                  'xgbclassifier__n_estimators': n_estimator}

    cv = StratifiedKFold(n_splits=5)
    pipeline = Pipeline([('under', random), ('xgbclassifier', model)])
    grid = GridSearchCV(estimator=pipeline, param_grid=param_grid, n_jobs=-1, cv=cv, scoring='roc_auc')
    # execute the grid search
    grid_result = grid.fit(X, y)
    # report the best configuration
    print(grid_result)
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    # report all configurations
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))


def grid_search_Reop_Smote(target,title, experience=False):
    print(title)
    print("--------------------------------------------------------------------")
    print()
    if experience is False:
        X = df_all.drop(
            ['SiteID', 'surgid', 'Complics', 'STSRCHOSPD', 'STSRCOM', 'Reoperation'], axis=1)
        y = df_all[target]
    else:
        X = df_all.drop(
            ['SiteID', 'surgid', 'Complics', 'STSRCHOSPD', 'STSRCOM', 'Reoperation',
             'HospID_Reop_CABG', 'HospID_total_CABG', 'surgyear', 'HospID_total_cardiac_surgery',
             'surgid_total_cardiac_surgery', 'surgid_total_CABG', 'surgid_Reop_CABG'], axis=1)
        y = df_all[target]

    counter = Counter(y)
    # estimate scale_pos_weight value
    estimate = counter[0] / counter[1]
    print('Estimate: %.3f' % estimate)
    print(counter[0])
    print(counter[1])

    model = XGBClassifier(objective='binary:logistic', eval_metric='logloss')
    random = RandomUnderSampler(sampling_strategy='majority')
    max_depths = [1, 2, 3, 5, 8, 10, 14, 18]
    learning_rates = [0.1, 0.05, 0.01]
    n_estimator = [50, 100, 150, 200, 300, 400]
    param_grid = {'xgbclassifier__max_depth': max_depths,
                  'xgbclassifier__learning_rate': learning_rates,
                  'xgbclassifier__n_estimators': n_estimator}

    cv = StratifiedKFold(n_splits=5)
    pipeline = Pipeline([('sample', SMOTE()), ('xgbclassifier', model)])
    grid = GridSearchCV(estimator=pipeline, param_grid=param_grid, n_jobs=-1, cv=cv, scoring='roc_auc')
    # execute the grid search
    grid_result = grid.fit(X, y)
    # report the best configuration
    print(grid_result)
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    # report all configurations
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))

def grid_search_Reop_Scale(target,title,experience=False):
    print(title)
    print("--------------------------------------------------------------------")
    print()
    if experience is False:
        X = df_all.drop(
            ['SiteID', 'surgid', 'Complics', 'STSRCHOSPD', 'STSRCOM', 'Reoperation'], axis=1)
        y = df_all[target]
    else:
        X = df_all.drop(
            ['SiteID', 'surgid', 'Complics', 'STSRCHOSPD', 'STSRCOM', 'Reoperation',
             'HospID_Reop_CABG', 'HospID_total_CABG', 'surgyear', 'HospID_total_cardiac_surgery',
             'surgid_total_cardiac_surgery', 'surgid_total_CABG', 'surgid_Reop_CABG'], axis=1)
        y = df_all[target]

    counter = Counter(y)
    # estimate scale_pos_weight value
    estimate = counter[0] / counter[1]
    print('Estimate: %.3f' % estimate)
    print(counter[0])
    print(counter[1])
    model = XGBClassifier(objective='binary:logistic', eval_metric='logloss')
    weights = [1, 5, 10, 25, 50, 75, 99, 100]
    param_grid = {'xgbclassifier__scale_pos_weight': weights}
    cv = StratifiedKFold(n_splits=5)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=cv, scoring='roc_auc')
    # execute the grid search
    grid_result = grid.fit(X, y)
    # report the best configuration
    print(grid_result)
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    # report all configurations
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))


def choose_script(model):
    if model == '1':
        grid_search_Reop_UnderSampling('STSRCOM', 'STSRCOM UnderSampling without experience',True)
    if model == '2':
        grid_search_Reop_UnderSampling('STSRCOM', 'STSRCOM UnderSampling with experience')

    if model == '3':
        grid_search_Reop_UnderSampling('STSRCHOSPD', 'STSRCHOSPD UnderSampling without experience',True)

    if model == '4':
        grid_search_Reop_UnderSampling('STSRCHOSPD', 'STSRCHOSPD UnderSampling with experience')

    if model == '5':
        grid_search_Reop_UnderSampling('Complics', 'Complics UnderSampling without experience', True)
    if model == '6':
        grid_search_Reop_UnderSampling('Complics', 'Complics UnderSampling with experience')

    if model == '7':
        grid_search_Reop_Smote('STSRCOM', 'STSRCOM SMOTE without experience',True)
    if model == '8':
        grid_search_Reop_Smote('STSRCOM', 'STSRCOM SMOTE with experience')

    if model == '9':
        grid_search_Reop_Smote('STSRCHOSPD', 'STSRCHOSPD SMOTE without experience',True)

    if model == '10':
        grid_search_Reop_Smote('STSRCHOSPD', 'STSRCHOSPD SMOTE with experience')

    if model == '11':
        grid_search_Reop_Smote('Complics', 'Complics SMOTE without experience',True)
    if model == '12':
        grid_search_Reop_Smote('Complics', 'Complics SMOTE with experience')


if __name__ == "__main__":
    print("test")
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='not', type=str)
    args = parser.parse_args()
    print (args.model)
    choose_script(args.model)