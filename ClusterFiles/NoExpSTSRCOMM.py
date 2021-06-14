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

df_all = pd.read_csv("/home/yaararum/Files/new data sum info surg and Hosp numeric values.csv")
path = "/home/yaararum/Files/model_outputs/"

df_all = df_all.replace({'STSRCHOSPD': {False: 0, True: 1}})
df_all = df_all.replace({'Complics': {False: 0, True: 1}})
df_all = df_all.replace({'Mortality': {False: 0, True: 1}})
df_all = df_all.replace({'STSRCOM': {False: 0, True: 1}})
df_all = df_all.replace({'PrCVInt': {False: 0, True: 1}})

print(Counter(df_all['STSRCHOSPD']))
print(Counter(df_all['STSRCOM']))

df_all.rename(columns={"EF<=35%": "EF_less_equal_35"}, inplace=True)

df_model_draft = df_all[
    ['HospID', 'HospID_Reop_CABG', 'HospID_total_CABG', 'surgyear', 'HospID_total_cardiac_surgery',
     'surgid', 'surgid_total_cardiac_surgery', 'surgid_total_CABG', 'surgid_Reop_CABG', 'SiteID', 'Age',
     'Gender', 'RaceCaucasian', 'RaceBlack', 'RaceOther', 'Ethnicity',
     'FHCAD', 'Diabetes', 'Hypertn', 'Dyslip', 'Dialysis', 'InfEndo', 'ChrLungD', 'ImmSupp', 'PVD', 'CreatLst',
     'PrevMI', 'Arrhythmia', 'PrCVInt', 'POCPCI', 'MedACEI', 'MedASA',
     'MedBeta', 'MedInotr', 'MedNitIV', 'MedSter', 'NumDisV', 'HDEF', 'VDInsufA', 'VDStenA', 'VDInsufM', 'VDStenM',
     'VDInsufT', 'VDStenT', 'Status', 'SmokingStatus', 'InsulinDiab',
     'ModSevereLungDis', 'PreCVAorTIAorCVD', 'RenFail', 'Angina', 'UnstableAngina', 'ClassNYHGroup',
     'ArrhythAtrFibFlutter', 'ArrhythOther', 'DualAntiPlat', 'MedHeparin', 'AntiCoag',
     'MedAntiplateltNoASA', 'NumDisV_ordinal', 'LeftMain', 'EF_less_equal_35', 'BMI',
     'Complics', 'STSRCHOSPD', 'STSRCOM', 'Reoperation']].copy()

# df_small = df_model_draft[df_model_draft['surgyear'].isin([2015])]
# print (df_small["HospID_total_cardiac_surgery"].unique())
df_t = df_model_draft[:5000]
X = df_model_draft.drop(
    ['HospID', 'SiteID', 'surgid', 'Complics', 'STSRCHOSPD', 'STSRCOM',
     'HospID_Reop_CABG', 'HospID_total_CABG', 'surgyear', 'HospID_total_cardiac_surgery',
     'surgid_total_cardiac_surgery', 'surgid_total_CABG', 'surgid_Reop_CABG'], axis=1)
y = df_model_draft['STSRCOM']  # La
print(X.isna().sum())
print(y.isna().sum())


def choose_script(model):
    if model == '1':
        print("model without experience with scale pos STSRCOM")
        print("--------------------------------------------------------------------")
        counter = Counter(y)
        # estimate scale_pos_weight value
        estimate = counter[0] / counter[1]
        print('Estimate: %.3f' % estimate)
        print(counter[0])
        print(counter[1])
        model = XGBClassifier(objective='binary:logistic', eval_metric='logloss')
        weights = [1, 5, 10, 25, 50, 75, 99, 100, 1000]
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

    if model == '2':
        print("model without experience with undersampling STSRCOM")
        print("--------------------------------------------------------------------")
        counter = Counter(y)
        # estimate scale_pos_weight value
        estimate = counter[0] / counter[1]
        print('Estimate: %.3f' % estimate)
        print(counter[0])
        print(counter[1])
        model = XGBClassifier(objective='binary:logistic', eval_metric='logloss')
        random = RandomUnderSampler(sampling_strategy=0.33)
        max_depths = [2, 3, 5, 8, 10, 14, 18]
        param_grid = {'xgbclassifier__max_depth': max_depths}
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
    if model == '3':
        print("model without experience with undersampling STSRCOM")
        print("--------------------------------------------------------------------")
        counter = Counter(y)
        # estimate scale_pos_weight value
        estimate = counter[0] / counter[1]
        print('Estimate: %.3f' % estimate)
        print(counter[0])
        print(counter[1])
        model = XGBClassifier(objective='binary:logistic', eval_metric='logloss')
        random = RandomUnderSampler(sampling_strategy=0.33)
        learning_rates = [0.1, 0.05, 0.01, 0.01]
        max_depths = [1,2]
        param_grid = {'xgbclassifier__max_depth': max_depths,
                      'xgbclassifier__learning_rate': learning_rates}
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
    if model == '4':
            print("model without experience with undersampling STSRCOM")
            print("--------------------------------------------------------------------")
            counter = Counter(y)
            # estimate scale_pos_weight value
            estimate = counter[0] / counter[1]
            print('Estimate: %.3f' % estimate)
            print(counter[0])
            print(counter[1])
            model = XGBClassifier(objective='binary:logistic', eval_metric='logloss')
            random = RandomUnderSampler(sampling_strategy=0.33)
            learning_rates = [0.1]
            max_depths = [2]
            n_estimator = [100, 200, 300, 400, 500]
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

    if model == '5':
        print("model without experience with Smote STSRCOM")
        print("--------------------------------------------------------------------")
        counter = Counter(y)
        # estimate scale_pos_weight value
        estimate = counter[0] / counter[1]
        print('Estimate: %.3f' % estimate)
        print(counter[0])
        print(counter[1])
        model = XGBClassifier(objective='binary:logistic', eval_metric='logloss')
        random = RandomUnderSampler(sampling_strategy=0.33)
        max_depths = [2, 3, 5, 8, 10, 14, 18]
        param_grid = {'xgbclassifier__max_depth': max_depths}
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
    if model == '6':
        print("model without experience with Smote STSRCOM")
        print("--------------------------------------------------------------------")
        counter = Counter(y)
        # estimate scale_pos_weight value
        estimate = counter[0] / counter[1]
        print('Estimate: %.3f' % estimate)
        print(counter[0])
        print(counter[1])
        model = XGBClassifier(objective='binary:logistic', eval_metric='logloss')
        random = RandomUnderSampler(sampling_strategy=0.33)
        learning_rates = [0.1, 0.05, 0.01, 0.01]
        max_depths = [18,22]
        param_grid = {'xgbclassifier__max_depth': max_depths,
                      'xgbclassifier__learning_rate': learning_rates}
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
    if model == '7':
            print("model without experience with Smote STSRCOM")
            print("--------------------------------------------------------------------")
            counter = Counter(y)
            # estimate scale_pos_weight value
            estimate = counter[0] / counter[1]
            print('Estimate: %.3f' % estimate)
            print(counter[0])
            print(counter[1])
            model = XGBClassifier(objective='binary:logistic', eval_metric='logloss')
            random = RandomUnderSampler(sampling_strategy=0.33)
            learning_rates = [0.1]
            max_depths = [22,26]
            n_estimator = [100, 200, 300, 400, 500]
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


if __name__ == "__main__":
    print("test")
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='not', type=str)
    args = parser.parse_args()
    print(args.model)
    choose_script(args.model)