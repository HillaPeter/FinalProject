import pandas as pd
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn import linear_model
import statsmodels.api as sm
import numpy as np
from scipy.stats import uniform, randint
from scipy import stats
from sklearn.datasets import load_breast_cancer, load_diabetes, load_wine
from sklearn.metrics import auc, accuracy_score, confusion_matrix, mean_squared_error
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, RandomizedSearchCV, train_test_split, \
    StratifiedKFold, RepeatedStratifiedKFold, GroupShuffleSplit
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler
import xgboost as xgb
import pandas as pd
import numpy as np
import os
import math
import timeit
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import *
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from imblearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, roc_curve
from imblearn.pipeline import Pipeline as imbpipeline
# from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings("ignore")

# df_2018_2019 = pd.read_csv("/mnt/nadavrap-students/STS/data/2018_2019.csv")
# print(df_2018_2019)
df_all = pd.read_csv("/tmp/pycharm_project_723/new data sum info surg and Hosp numeric values.csv")
# df_all = pd.read_csv("/tmp/pycharm_project_723/imputed_data_with_numerical_values.csv")
# df_all2 = pd.read_csv("/tmp/pycharm_project_723/imputed_data2.csv")
#
print(df_all.columns.tolist())




df_all = df_all.replace({'Complics':{False:0, True:1}})
df_all = df_all.replace({'Mortalty':{False:0, True:1}})
df_all = df_all.replace({'PrCVInt':{False:0, True:1}})
df_all = df_all.replace({'Mortality':{False:0, True:1}})
df_all = df_all.replace({'Mt30Stat':{'Alive':0, 'Dead':1, np.nan:2}})

# df_all = df_all.replace({'Reoperation':{'First Time':0, 'Reoperation':1}})
# df_all.rename(columns={"EF<=35%": "EF_less_equal_35"}, inplace=True)
# print (df_all['HospID_total_cardiac_surgery'].isna().sum())
# # df_all['HospID_total_cardiac_surgery'].str.strip(',').astype(float)
# # df_all["HospID_total_cardiac_surgery"] = pd.to_numeric(df_all["HospID_total_cardiac_surgery"])
# # df_all["HospID_total_cardiac_surgery"] = df_all["HospID_total_cardiac_surgery"].str.strip(', ')
# df_all["HospID_total_cardiac_surgery"] = df_all["HospID_total_cardiac_surgery"].astype(float)
# print (df_all['HospID_total_cardiac_surgery'].isna().sum())
# # df_test= df_all[
# #         ['HospID','HospID_total_cardiac_surgery', 'HospID_Reop_CABG', 'HospID_total_CABG', 'surgyear',
# #          'surgid', 'surgid_total_cardiac_surgery','surgid_total_CABG', 'surgid_Reop_CABG', 'SiteID',  'Complics', 'Mortalty']].copy()
# #
# # mask = df_all['surgyear'] == 2019 and df_all['surgyear'] == 2018
# # df_2019 = df_all[mask]
# # df_2019.to_csv("2018 2019.csv")
# # print (df_all.columns.tolist())
# # print (df_all.head(10))
# # # df_all[:50].to_csv("numeric_df_after changes.csv")
# df_model_draft = df_all[
#         ['HospID', 'HospID_Reop_CABG', 'HospID_total_CABG', 'surgyear','HospID_total_cardiac_surgery'
#          'surgid', 'surgid_total_cardiac_surgery','surgid_total_CABG', 'surgid_Reop_CABG', 'SiteID',  'Age',
#          'Gender', 'RaceCaucasian', 'RaceBlack', 'RaceOther', 'Ethnicity',
#          'FHCAD', 'Diabetes', 'Hypertn', 'Dyslip', 'Dialysis', 'InfEndo', 'ChrLungD', 'ImmSupp', 'PVD', 'CreatLst',
#          'PrevMI', 'Arrhythmia', 'PrCVInt', 'POCPCI', 'MedACEI', 'MedASA',
#          'MedBeta', 'MedInotr', 'MedNitIV', 'MedSter', 'NumDisV', 'HDEF', 'VDInsufA', 'VDStenA', 'VDInsufM', 'VDStenM',
#          'VDInsufT', 'VDStenT', 'Status', 'SmokingStatus', 'InsulinDiab',
#          'ModSevereLungDis', 'PreCVAorTIAorCVD', 'RenFail', 'Angina', 'UnstableAngina', 'ClassNYHGroup',
#          'ArrhythAtrFibFlutter', 'ArrhythOther', 'DualAntiPlat', 'MedHeparin', 'AntiCoag',
#          'MedAntiplateltNoASA', 'NumDisV_ordinal', 'LeftMain', 'EF_less_equal_35', 'BMI',
#          'Complics', 'Mortality', 'Reoperation']].copy()
#
# df_small = df_model_draft[df_model_draft['surgyear'].isin([2015])]
# print (df_small["HospID_total_cardiac_surgery"].unique())

# X = df_small.drop(
#     ['HospID', 'SiteID', 'surgid', 'Complics', 'Mortality'], axis=1)
# y = df_small['Mortality']  # La
# print (X.isna().sum())
# print(y.isna().sum())

labels = ['TN', 'FP', 'FN', 'TP']
categories = ['Negative', 'Positive']
N_split=5

def Make_Confusion_Matrix(cf,
                          group_names=None,
                          categories='auto',
                          count=True,
                          percent=True,
                          cbar=True,
                          xyticks=True,
                          xyplotlabels=True,
                          sum_stats=True,
                          figsize=None,
                          cmap='Pastel1',
                          title=None,
                          y_pred=None,
                          y_test=None):

    # CODE TO GENERATE TEXT INSIDE EACH SQUARE
    blanks = ['' for i in range(cf.size)]

    if group_names and len(group_names) == cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
    else:
        group_counts = blanks

    if percent:
        group_percentages = ["{0:.2%}".format(value) for value in cf.flatten() / np.sum(cf)]
    else:
        group_percentages = blanks

    box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels, group_counts, group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0], cf.shape[1])

    # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
    if sum_stats:
        # Accuracy is sum of diagonal divided by total observations
        accuracy = np.trace(cf) / float(np.sum(cf))

        # if it is a binary confusion matrix, show some more stats
        if len(cf) == 2:
            # Metrics for Binary Confusion Matrices
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            specificity = tn / (tn+fp)
            precision = cf[1, 1] / sum(cf[:, 1])
            recall = cf[1, 1] / sum(cf[1, :])
            f1_score_test = f1_score(y_test, y_pred,average='macro')
            stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nSensitivity={:0.3f}\nF1 Score={:0.3f}\nSpecificity={:0.3f}".format(accuracy, precision, recall, f1_score_test , specificity)
        else:
            stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
    else:
        stats_text = ""

    # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
    fig = plt.figure()
    sns.set(style="white")

    sns.heatmap(cf, annot=box_labels, cmap=cmap, fmt='', cbar=cbar, xticklabels=categories, yticklabels=categories)
    if xyplotlabels:
        plt.ylabel('True Class')
        plt.xlabel('Predicted Class' + stats_text)
    else:
        plt.xlabel(stats_text)

    if title:
        plt.title(title)

    plt.show()
    return {'F1 Score': f1_score_test, 'Accuracy': accuracy, 'Sensitivity': recall, 'Specificity': specificity}


def hyper_paramitize_scale_gridSearch():
    counter = Counter(y)
    # estimate scale_pos_weight value
    estimate = counter[0] / counter[1]
    print('Estimate: %.3f' % estimate)
    print (counter[0])
    print(counter[1])
    model = XGBClassifier(objective='binary:logistic', eval_metric='logloss')
    random = RandomUnderSampler(sampling_strategy=0.33)
    # define grid
    # weights = [1,3, 10, 25,30, 50, 75, 99, 100]
    #param_grid = dict(scale_pos_weight=weights)
    #param_grid= {'xgbclassifier__scale_pos_weight': weights}
    learning_rates = [0.1, 0.05, 0.01]
    max_depths = [1,2,3,5,8,10]
    param_grid = {'xgbclassifier__max_depth': max_depths,
                  'xgbclassifier__learning_rate': learning_rates}
    # define evaluation procedure
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=1)
    # define grid search
    pipeline = Pipeline([('under',random ), ('xgbclassifier', model)])

    grid = GridSearchCV(estimator=pipeline, param_grid=param_grid, n_jobs=-1, cv=cv, scoring='roc_auc')
    # execute the grid search
    grid_result = grid.fit(X, y)
    # report the best configuration
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    # report all configurations
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))



# hyper_paramitize_scale_gridSearch()



def smote():
    pipeline = imbpipeline(steps=[['smote', RandomOverSampler()],
                                  ['classifier', XGBClassifier(objective='binary:logistic', eval_metric='logloss')]])

    stratified_kfold = StratifiedKFold(n_splits=10,
                                       shuffle=True,
                                       random_state=11)
    max_depths = [2 ** x for x in range(1, 7)]
    num_estimators = [10, 20, 30] + list(range(45, 100, 5))
    learning_rates = [0.1, 0.05, 0.01]
    param_grid = {'classifier__max_depth': max_depths,
                  'classifier__n_estimators': num_estimators,
                  'classifier__learning_rate': learning_rates}
    grid_search = GridSearchCV(estimator=pipeline,
                               param_grid=param_grid,
                               scoring='roc_auc',
                               cv=stratified_kfold,
                               n_jobs=-1)

    grid_search.fit(X, y)
    print(grid_search)
    print(grid_search.best_estimator_)
    print(grid_search.best_params_)
    cv_score = grid_search.best_score_
    test_score = grid_search.score(X, y)
    print(f'Cross-validation score: {cv_score}\nTest score: {test_score}')

# smote()

def hyperparameterCVoptimization():
    # parameters = {
    #         "learning_rate": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
    #         "max_depth": [3, 4, 5, 6, 8, 10, 12, 15],
    #         "gamma": [0.0, 0.1, 0.2, 0.3, 0.4],
    #         "colsample_bytree": [0.3, 0.4, 0.5, 0.7]
    #     }
    X = df_small.drop(
        ['HospID', 'SiteID', 'surgid', 'Complics', 'Mortality'], axis=1)
    y = df_small['Mortality']  # La
    params = {
        'max_depth': range(2, 10, 1),
        # 'n_estimators': range(60, 220, 40),
        # 'learning_rate': [0.1, 0.01, 0.05]
    }
    i=1
    kf = StratifiedKFold(n_splits=5,random_state=1,shuffle=True)
    for train_index,test_index in kf.split(np.zeros(len(y)), y):
         print('\n{} of kfold {}'.format(i,kf.n_splits))
         xtr,xvl = X.iloc[train_index],X.iloc[test_index]
         ytr,yvl = y.iloc[train_index],y.iloc[test_index]
         model = GridSearchCV(XGBClassifier(objective='binary:logistic', eval_metric='logloss'), param_grid=params, cv=5, scoring= 'roc_auc')
         model.fit(xtr, ytr)
         print (model.best_params_)
         pred=model.predict(xvl)
         print('accuracy_score',accuracy_score(yvl,pred))
         print (classification_report(yvl,pred))
         y_pred = model.predict_proba(xvl)[:, 1]
         print ('roc-auc',roc_auc_score(yvl, y_pred) )
         i+=1
         print ("==========================================================")


# hyperparameterCVoptimization()

def cvsmote():
    X = df_small.drop(
        ['HospID', 'SiteID', 'surgid', 'Complics', 'Mortality'], axis=1)
    y = df_small['Mortality']

    steps = [('over', SMOTE()), ('model', XGBClassifier(objective='binary:logistic', eval_metric='logloss'))]
    pipeline = Pipeline(steps=steps)
    # evaluate pipeline
    for scoring in ["accuracy", "roc_auc"]:
        cv = StratifiedKFold(n_splits=10, random_state=0)
        scores = cross_val_score(pipeline, X, y, scoring=scoring, cv=cv, n_jobs=-1)
        print("Model", scoring, " mean=", scores.mean(), "stddev=", scores.std())


def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))

def test_df(df):
    for col in df:
      print (  df['col1'].equals(df['col2']))


# Driver Code

def check_split_differnce():
    df_train_hosp = pd.DataFrame()
    df_test_hosp = pd.DataFrame()
    for i in range(9):
        TRAIN_TEST_SPLIT_PERC = 0.8
        uniques = df_all["HospID"].unique()
        sep = int(len(uniques) * TRAIN_TEST_SPLIT_PERC)
        df = df_all.sample(frac=1).reset_index(drop=True)  # For shuffling your data
        train_ids, test_ids = uniques[:sep], uniques[sep:]
        train_df = df_all.loc[train_ids]
        test_df = df_all.loc[test_ids]
        print(intersection(train_ids, test_ids))
        df_train_hosp[i+1] = train_df['HospID']
        df_test_hosp[i+1] = test_df['HospID']

    df_train_hosp.to_csv("train same split.csv")
    df_test_hosp.to_csv("test same split.csv")


def check_split_differnce_shuffle():
    df_train_hosp = pd.DataFrame()
    df_test_hosp = pd.DataFrame()


    groups = df_all['HospID']
    print(groups.shape)
    print(groups.unique())
    gss = GroupShuffleSplit(n_splits=10, train_size=.8, random_state=42)
    gss.get_n_splits()
    i = 1
    X = df_all.drop(
        ['SiteID', 'surgid', 'Complics', 'STSRCHOSPD', 'STSRCOM'], axis=1)
    y = df_all['Mortality']
    for train_idx, test_idx in gss.split(X, y, groups):
        print("TRAIN:", train_idx, "TEST:", test_idx)

        print(X.columns.tolist())
        train_df= X.loc[train_idx]
        test_df= X.loc[train_idx]
        df_train_hosp[i + 1] = train_df['HospID']
        df_test_hosp[i + 1] = test_df['HospID']

    df_train_hosp.to_csv("train same shuffle split.csv")
    df_test_hosp.to_csv("test same shuffle split.csv")

# check_split_differnce()
# check_split_differnce_shuffle()
def splitbyid():
    # TRAIN_TEST_SPLIT_PERC = 0.8
    # uniques = df_all["HospID"].unique()
    # sep = int(len(uniques) * TRAIN_TEST_SPLIT_PERC)
    # df = df_all.sample(frac=1).reset_index(drop=True)  # For shuffling your data
    # train_ids, test_ids = uniques[:sep], uniques[sep:]
    # print (intersection(train_ids, test_ids))
    # train_df, test_df = df[df.HospID.isin(train_ids)], df[df.HospID.isin(test_ids)]
    #
    # print("\nTRAIN DATAFRAME\n", train_df.shape)
    # print("\nTEST DATAFRAME\n", test_df.shape)

    groups = df_all['HospID']
    print(groups.shape)
    print(groups.unique())
    gss = GroupShuffleSplit(n_splits=2, train_size=.8, random_state=8)
    gss.get_n_splits()
    i = 1
    X = df_all.drop(
        ['SiteID', 'surgid', 'Complics', 'STSRCHOSPD', 'STSRCOM'], axis=1)
    y = df_all['Mortality']
    for train_idx, test_idx in gss.split(X, y, groups):
        # print("TRAIN:", train_idx, "TEST:", test_idx)

        # print(X.columns.tolist())
        train_df= X.loc[train_idx]
        test_df= X.loc[train_idx]
        t = train_df["HospID"].unique()
        f = test_df["HospID"].unique()
        print (train_df.shape)
        print(test_df.shape)
        return t,f


def sublist1(test_list1, test_list2):
    # ls1 = [element for element in lst1 if element in lst2]
    # ls2 = [element for element in lst2 if element in lst1]
    # return ls1 == ls2
    # sorting both the lists
    # using == to check if
    # lists are equal
    print(len(test_list1))
    print(len(test_list2))
    if set(test_list1) == set(test_list2):
        print("The lists are identical")
    else:
        print("The lists are not identical")

print ("1")
l1,l2 = splitbyid()
print (l1)
print (l2)
print("===================================================================")
print("2")
l3, l4 = splitbyid()
print("===================================================================")
print("3")
l5, l6 = splitbyid()
print("===================================================================")
print ("intersection")
print(sublist1(l1,l3))
print(sublist1(l2, l4))
print(sublist1(l5, l3))
print(sublist1(l1, l2))

print(sublist1([1,2,3], [3,5,4]))
# def splitbyid():
#     TRAIN_TEST_SPLIT_PERC = 0.8
#     uniques = df_all["HospID"].unique()
#     sep = int(len(uniques) * TRAIN_TEST_SPLIT_PERC)
#     df = df_all.sample(frac=1).reset_index(drop=True)  # For shuffling your data
#     train_ids, test_ids = uniques[:sep], uniques[sep:]
#     print (intersection(train_ids, test_ids))
#     train_df, test_df = df[df.HospID.isin(train_ids)], df[df.HospID.isin(test_ids)]
#
#     print("\nTRAIN DATAFRAME\n", train_df.shape)
#     print("\nTEST DATAFRAME\n", test_df.shape)
#     print(train_df["HospID"].unique())
#     print(test_df["HospID"].unique())
    # X_train = train_df.drop(
    #     ['HospID', 'SiteID', 'surgid', 'Complics', 'Mortality'], axis=1)
    # y_train = train_df['Mortality']
    # X_test = test_df.drop(
    #     ['HospID', 'SiteID', 'surgid', 'Complics', 'Mortality'], axis=1)
    # y_test = test_df['Mortality']
    #
    # undersample = RandomUnderSampler(sampling_strategy=0.33)  # 'majority'
    # # fit and apply the transform
    # X_over, y_over = undersample.fit_resample(X_train, y_train)
    # # summarize class distribution
    # print("after under sampling")
    # counter = Counter(y_over)
    # print(counter)
    # estimate = counter[0] / counter[1]
    # print('Estimate: %.3f' % estimate)
    # model = XGBClassifier(objective='binary:logistic', eval_metric='logloss',max_depth=5,learning_rate=0.1,scale_pos_weight=estimate)
    # model.fit(X_over,y_over)
    #
    #
    # pred = model.predict(X_test)
    # print (" Model with scale")
    # print('accuracy_score', accuracy_score(y_test, pred))
    # print(classification_report(y_test, pred))
    # y_pred = model.predict_proba(X_test)[:, 1]
    # print('roc-auc', roc_auc_score(y_test, y_pred))
    # cm = confusion_matrix(y_test, y_pred)
    # Make_Confusion_Matrix(cm, categories=categories, cmap='PuRd',
    #                       title='Mortality prediction with expereince and scale', group_names=labels, y_pred=y_pred,
    #                       y_test=y_test)
    #
    # print(" ====================================================")
    # print()
    # print(" Model without scale")
    # X_over2, y_over2 = undersample.fit_resample(X_train, y_train)
    # # summarize class distribution
    # print("after under sampling")
    # print(Counter(y_over2))
    #
    # model2 = XGBClassifier(objective='binary:logistic', eval_metric='logloss', max_depth=5, learning_rate=0.1)
    # model2.fit(X_over2, y_over2)
    #
    # pred = model2.predict(X_test)
    # print('accuracy_score', accuracy_score(y_test, pred))
    # print(classification_report(y_test, pred))
    # y_pred = model2.predict_proba(X_test)[:, 1]
    # print('roc-auc', roc_auc_score(y_test, y_pred))
    # cm = confusion_matrix(y_test, y_pred)
    # Make_Confusion_Matrix(cm, categories=categories,
    #                       title='Mortality prediction withe xpreience and without scale', group_names=labels, y_pred=y_pred,
    #                       y_test=y_test)

# splitbyid()
#cvsmote()
# def K_Fold_Split(g_search_model, model_name,  df, Y,color='YlGnBu',n=5):
#     stratified_kfold = StratifiedKFold(n_splits=n, shuffle=True, random_state=4)
#     metrics_score = []
#     best_params_evaluation_test = []
#     best_models = []
#     start_KFold = timeit.default_timer()
#     print("## Start {} Fold on Model '{}' ##".format(n, model_name))
#     k = 1
#     for train_index, test_index in stratified_kfold.split(np.zeros(len(Y)), Y):
#         print("\tStart Fold Number {}".format(k))
#         train_valid_X = df.iloc[train_index]
#         test_X = df.iloc[test_index]
#
#         train_valid_Y = Y.iloc[train_index]
#         test_Y = Y.iloc[test_index]
#
#         lbls_dict = {'Negative': 0,
#                      'Positive': 1}
#
#         start_nested_kfold = timeit.default_timer()
#         g_search_model.fit(train_valid_X, train_valid_Y)
#         stop_nested_kfold = timeit.default_timer()
#         total_nested_kfold = stop_nested_kfold - start_nested_kfold
#         print("\tEnd Nested KFold Process in Time : {} Sec".format(round(total_nested_kfold, 2)))
#         best_params_evaluation_test.append(g_search_model.best_params_)
#         best_model = g_search_model.best_estimator_
#         # Evaluate(best_model, train_valid_X, train_valid_Y,lbls_dict)
#
#         pred_Y = best_model.predict(test_X)
#         cm = confusion_matrix(test_Y, pred_Y)
#         metrics_test = Make_Confusion_Matrix(cm, categories=categories, cmap=color,
#                                              title='SARS-Cov-2 exam result Classification on Test Set',
#                                              group_names=labels)
#         metrics_score.append(metrics_test)
#         best_models.append(best_model)
#         roc = roc_auc_score(test_Y,best_model.predict_proba(test_X))
#         print ("==========================================================================================================")
#         print ('K-Fold {} ROC score {}'.format(k,roc))
#         print ("==========================================================================================================")
#
#         k += 1
#     stop_KFold = timeit.default_timer()
#     total_KFold = stop_KFold - start_KFold
#     print("Ended {} Fold Model : {}. Time : {} Sec".format(n, model_name, round(total_KFold, 2)))
#     return metrics_score, best_params_evaluation_test,best_models

# group = df_small['HospID'].unique()
# print(group.shape)
# X = df_small.drop(
#     ['SiteID', 'surgid', 'Complics', 'Mortality'], axis=1)
# # 'HospID_total_cardiac_surgery', 'HospID_Reop_CABG', 'HospID_total_CABG', 'surgyear',
# # 'surgid_total_cardiac_surgery','surgid_total_CABG', 'surgid_Reop_CABG'], axis=1)
# y = df_small['Mortality']
# gss = GroupShuffleSplit(n_splits=2, train_size=.7, random_state=42)
# print(gss.get_n_splits())
#
# for train_idx, test_idx in gss.split(X, y, group):
#     print("TRAIN:", df_small[train_idx], "TEST:", df_small[test_idx])
def Make_Confusion_Matrix(cf,
                          group_names=None,
                          categories='auto',
                          count=True,
                          percent=True,
                          cbar=True,
                          xyticks=True,
                          xyplotlabels=True,
                          sum_stats=True,
                          figsize=None,
                          cmap='Pastel1',
                          title=None,
                          y_pred=None,
                          y_test=None):

    # CODE TO GENERATE TEXT INSIDE EACH SQUARE
    blanks = ['' for i in range(cf.size)]

    if group_names and len(group_names) == cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
    else:
        group_counts = blanks

    if percent:
        group_percentages = ["{0:.2%}".format(value) for value in cf.flatten() / np.sum(cf)]
    else:
        group_percentages = blanks

    box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels, group_counts, group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0], cf.shape[1])

    # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
    if sum_stats:
        # Accuracy is sum of diagonal divided by total observations
        accuracy = np.trace(cf) / float(np.sum(cf))

        # if it is a binary confusion matrix, show some more stats
        if len(cf) == 2:
            # Metrics for Binary Confusion Matrices
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            specificity = tn / (tn+fp)
            precision = cf[1, 1] / sum(cf[:, 1])
            recall = cf[1, 1] / sum(cf[1, :])
            f1_score_test = f1_score(y_test, y_pred,average='macro')
            stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nSensitivity={:0.3f}\nF1 Score={:0.3f}\nSpecificity={:0.3f}".format(accuracy, precision, recall, f1_score_test , specificity)
        else:
            stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
    else:
        stats_text = ""

    # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
    fig = plt.figure()
    sns.set(style="white")

    sns.heatmap(cf, annot=box_labels, cmap=cmap, fmt='', cbar=cbar, xticklabels=categories, yticklabels=categories)
    if xyplotlabels:
        plt.ylabel('True Class')
        plt.xlabel('Predicted Class' + stats_text)
    else:
        plt.xlabel(stats_text)

    if title:
        plt.title(title)

    plt.show()
    return {'F1 Score': f1_score_test, 'Accuracy': accuracy, 'Sensitivity': recall, 'Specificity': specificity}

def Create_XGBoost_Model():
    learning_rates = [0.1,0.05,0.01]
    num_estimators = [10,20,30] + list(range(45,100,5))
    max_depths = [2**x for x in range(1,7)]
    grid = {'xgbclassifier__learning_rate': learning_rates,
            'xgbclassifier__n_estimators': num_estimators,
            'xgbclassifier__max_depth': max_depths}
    xgb_model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss')
    cv_kfold = KFold(n_splits=5 , shuffle = True , random_state = 4)
    pipeline = Pipeline([('sample', SMOTE()), ('xgbclassifier', xgb_model)])
    xgb_model_grid_search = GridSearchCV(estimator = pipeline, param_grid = grid,scoring='roc_auc', cv = cv_kfold, n_jobs = -1, verbose = 4)
    return xgb_model_grid_search


# def K_Fold_Split(model_name, color='YlGnBu', df=X, n=5):
#     stratified_kfold = StratifiedKFold(n_splits=n, shuffle=True, random_state=4)
#     metrics_score = []
#     best_params_evaluation_test = []
#     best_models = []
#     start_KFold = timeit.default_timer()
#     print("## Start {} Fold on Model '{}' ##".format(n, model_name))
#     k = 1
#     for train_index, test_index in stratified_kfold.split(np.zeros(len(Y)), Y):
#         g_search_model = Create_XGBoost_Model()
#         print("\tStart Fold Number {}".format(k))
#         train_valid_X = df.iloc[train_index]
#         test_X = df.iloc[test_index]
#
#         train_valid_Y = Y.iloc[train_index]
#         test_Y = Y.iloc[test_index]
#
#         lbls_dict = {'Negative': 0,
#                      'Positive': 1}
#
#         start_nested_kfold = timeit.default_timer()
#         g_search_model.fit(train_valid_X.values, train_valid_Y)
#         stop_nested_kfold = timeit.default_timer()
#         total_nested_kfold = stop_nested_kfold - start_nested_kfold
#         print("\tEnd Nested KFold Process in Time : {} Sec".format(round(total_nested_kfold, 2)))
#         best_params_evaluation_test.append(g_search_model.best_params_)
#         best_model = g_search_model.best_estimator_
#
#         pred_Y = best_model.predict(test_X.values)
#         cm = confusion_matrix(test_Y, pred_Y)
#         metrics_test = Make_Confusion_Matrix(cm, categories=categories, cmap=color,
#                                              title='SARS-Cov-2 exam result Classification on Test Set',
#                                              group_names=labels, y_pred=pred_Y, y_test=test_Y)
#
#         auc = roc_auc_score(test_Y, best_model.predict_proba(test_X.values)[:, 1])
#         metrics_test['AUROC'] = auc
#         metrics_score.append(metrics_test)
#         best_models.append(best_model)
#         k += 1
#         print()
#     stop_KFold = timeit.default_timer()
#     total_KFold = stop_KFold - start_KFold
#     print("Ended {} Fold Model : {}. Time : {} Sec".format(n, model_name, round(total_KFold, 2)))
#     return metrics_score, best_params_evaluation_test

def Compare_K_Fold_Model_XGB(metrics_tests, best_params_xgb_5Fold):
    index_rows = ['Max Depth','N Estimators','Learning Rate','F1 Score','Accuracy','Sensitivity','Specificity','AUROC']
    cols = ['Fold {}'.format(i+1) for i in range(5)]
    XGBoost_Comperison = pd.DataFrame(columns=cols , index=index_rows)
    for i in range(5):
        matrics = metrics_tests[i]
        md_p = best_params_xgb_5Fold[i]['xgbclassifier__max_depth']
        ne_p = best_params_xgb_5Fold[i]['xgbclassifier__n_estimators']
        lr_p = best_params_xgb_5Fold[i]['xgbclassifier__learning_rate']
        XGBoost_Comperison.loc[ 'Max Depth' , 'Fold {}'.format(i+1)] = md_p
        XGBoost_Comperison.loc[ 'N Estimators' , 'Fold {}'.format(i+1)] = ne_p
        XGBoost_Comperison.loc[ 'Learning Rate' , 'Fold {}'.format(i+1)] = lr_p
        for matric in matrics:
            XGBoost_Comperison.loc[ matric , 'Fold {}'.format(i+1)] = matrics[matric]
    return XGBoost_Comperison


# metrics_tests, best_params_xgb_5Fold = K_Fold_Split("XGBoost")
#
# XGB_1 = Compare_K_Fold_Model_XGB(metrics_tests, best_params_xgb_5Fold)
#
# print(XGB_1)
