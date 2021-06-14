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
    StratifiedKFold, RepeatedStratifiedKFold
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
import warnings
warnings.filterwarnings("ignore")

df_2018_2019 = pd.read_csv("/mnt/nadavrap-students/STS/data/2018_2019.csv")
print(df_2018_2019)
# df_all2 = pd.read_csv("/mnt/nadavrap-students/STS/data/imputed_data2.csv")
# df_all = pd.read_csv("/tmp/pycharm_project_723/imputed_data_with_numerical_values.csv")
df_all = pd.read_csv("/tmp/pycharm_project_723/imputed_data_with_float_values_glmm.csv")


df_all = df_all.replace({'Complics':{False:0, True:1}})
df_all = df_all.replace({'Mortalty':{False:0, True:1}})
df_all = df_all.replace({'PrCVInt':{False:0, True:1}})
df_all = df_all.replace({'Mortality':{False:0, True:1}})
df_all = df_all.replace({'Mt30Stat':{'Alive':0, 'Dead':1, np.nan:2}})

df_all = df_all.replace({'Reoperation':{'First Time':0, 'Reoperation':1}})
df_all.rename(columns={"EF<=35%": "EF_less_equal_35"}, inplace=True)
df_all['HospID_total_cardiac_surgery'] = df_all['HospID_total_cardiac_surgery'].str.replace(',', '').astype(float)



# df_test= df_all[
#         ['HospID','HospID_total_cardiac_surgery', 'HospID_Reop_CABG', 'HospID_total_CABG', 'surgyear',
#          'surgid', 'surgid_total_cardiac_surgery','surgid_total_CABG', 'surgid_Reop_CABG', 'SiteID',  'Complics', 'Mortalty']].copy()
#
# mask = df_all['surgyear'] == 2019 and df_all['surgyear'] == 2018
# df_2019 = df_all[mask]
# df_2019.to_csv("2018 2019.csv")
# print (df_all.columns.tolist())
# print (df_all.head(10))
# df_all[:50].to_csv("numeric_df_after changes.csv")
df_model_draft = df_all[
        ['HospID','HospID_total_cardiac_surgery', 'HospID_Reop_CABG', 'HospID_total_CABG', 'surgyear',
         'surgid', 'surgid_total_cardiac_surgery','surgid_total_CABG', 'surgid_Reop_CABG', 'SiteID',  'Age',
         'Gender', 'RaceCaucasian', 'RaceBlack', 'RaceOther', 'Ethnicity',
         'FHCAD', 'Diabetes', 'Hypertn', 'Dyslip', 'Dialysis', 'InfEndo', 'ChrLungD', 'ImmSupp', 'PVD', 'CreatLst',
         'PrevMI', 'Arrhythmia', 'PrCVInt', 'POCPCI', 'MedACEI', 'MedASA',
         'MedBeta', 'MedInotr', 'MedNitIV', 'MedSter', 'NumDisV', 'HDEF', 'VDInsufA', 'VDStenA', 'VDInsufM', 'VDStenM',
         'VDInsufT', 'VDStenT', 'Status', 'SmokingStatus', 'InsulinDiab',
         'ModSevereLungDis', 'PreCVAorTIAorCVD', 'RenFail', 'Angina', 'UnstableAngina', 'ClassNYHGroup',
         'ArrhythAtrFibFlutter', 'ArrhythOther', 'DualAntiPlat', 'MedHeparin', 'AntiCoag',
         'MedAntiplateltNoASA', 'NumDisV_ordinal', 'LeftMain', 'EF_less_equal_35', 'BMI',
         'Complics', 'Mortality', 'Reoperation']].copy()

# print (df_all['Mt30Stat'].unique())
# counter = Counter(df_all['Mt30Stat'])
# print(counter)
# df1 = df_all.groupby(['surgyear'])['Mt30Stat'].apply(lambda x: (x == 1).sum()).reset_index(name='Dead')
# df2 = df_all.groupby(['surgyear'])['surgyear'].count().reset_index(name='total')
# df3 = df_all.groupby(['surgyear'])['Mt30Stat'].apply(lambda x: (x == 0).sum()).reset_index(name='Alive')
# df6 = df_all.groupby(['surgyear'])['Mt30Stat'].apply(lambda x: (x == 2).sum()).reset_index(name='Nan')
# 
# df4 = pd.merge(df2, df1, on='surgyear', how='outer')
# df5 = pd.merge(df4, df3, on='surgyear', how='outer')
# df7 = pd.merge(df5, df6, on='surgyear', how='outer')
# print (df7.head(10))
# df7.to_csv("summary of years surg Mt30Stat.csv")
# 
# print(df_model_draft['Mortality'].value_counts())
# print(df_model_draft['Complics'].value_counts())


# df_train = df_model_draft[df_model_draft['surgyear'].isin([2010,2011,2012])]
df_train = df_model_draft[df_model_draft['surgyear'].isin([2010,2011,2012,2013,2014,2015,2016])]
df_test = df_model_draft[df_model_draft['surgyear'].isin([2017])]
print(df_train['surgyear'].value_counts())
print(df_test['surgyear'].value_counts())

def display_scores(scores):
    print("Scores: {0}\nMean: {1:.3f}\nStd: {2:.3f}".format(scores, np.mean(scores), np.std(scores)))

def report_best_scores(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


def HyperParameter(df):
    X = df.drop(['id', 'artist_id', 'album_title', 'is_gold'], axis=1)
    y = df['is_gold']

    xgb_model = xgb.XGBRegressor()

    params = {
        "colsample_bytree": uniform(0.7, 0.3),
        "gamma": uniform(0, 0.5),
        "learning_rate": uniform(0.03, 0.3),  # default 0.1
        "max_depth": randint(2, 6),  # default 3
        "n_estimators": randint(100, 150),  # default 100
        "subsample": uniform(0.6, 0.4)
    }

    search = RandomizedSearchCV(xgb_model, param_distributions=params, random_state=42, n_iter=200, cv=3, verbose=1,
                                n_jobs=1, return_train_score=True)

    search.fit(X, y)

    report_best_scores(search.cv_results_, 1)

def feature_importance(model, df,X,y,colorhist,cmapcolor):
    from matplotlib import pyplot

    plt.figure(figsize=(12, 4))
    feat_importances = pd.Series(model.feature_importances_, index=X.columns)
    featureImpList= []
    for feat, importance in zip(X.columns, model.feature_importances_):
        temp = [feat, importance * 100]
        featureImpList.append(temp)

    fT_df = pd.DataFrame(featureImpList, columns=['Feature', 'Importance'])
    print(fT_df.sort_values('Importance', ascending=False))

    feat_importances.nlargest(17).plot.bar(rot=0, color=colorhist, stacked=True)  # (kind='barh',rot=0,color='skyblue')
    plt.xticks(rotation=90)

    plt.ylabel('Importance')
    plt.xlabel('Variable')
    plt.title('Feature Importances')
    plt.show()

    # print (feat_importances.keys())
    #==================================pearson correlation===================
    X_new = df[feat_importances.nlargest(17).keys()]

    # print (X_new.head())
    corr = X_new.copy()
    corr['Mortality'] = y
    correlation = corr.corr().corr(method='pearson')

    fig, ax = plt.subplots(figsize=(14, 14))
    ax.set_title('Pearson correlation for variables')
    sns.heatmap(correlation,
                xticklabels=correlation.columns,
                yticklabels=correlation.columns,
                cmap= cmapcolor, #'RdPu',
                annot=True,
                fmt='.3f',
                linewidth=0.5, ax=ax,
                annot_kws={"fontsize": 8})
    plt.show()

#code for beautiful confusion matrix
def make_confusion_matrix(cf,
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
                          title=None):

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
            precision = cf[1, 1] / sum(cf[:, 1])
            recall = cf[1, 1] / sum(cf[1, :])
            f1_score = 2 * precision * recall / (precision + recall)
            stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                accuracy, precision, recall, f1_score)
        else:
            stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
    else:
        stats_text = ""

    # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
    fig = plt.figure()
    sns.set(style="white")

    sns.heatmap(cf, annot=box_labels, cmap=cmap, fmt='', cbar=cbar, xticklabels=categories, yticklabels=categories)
    if xyplotlabels:
        plt.ylabel('True label')
        plt.xlabel('Predicted label' + stats_text)
    else:
        plt.xlabel(stats_text)

    if title:
        plt.title(title)

    # plt.show()
    return plt

def make_roc_auc_curve(y_test,probas, title):
    from sklearn.metrics import roc_curve, auc
    # get false and true positive rates
    fpr, tpr, thresholds = roc_curve(y_test, probas[:, 0], pos_label=0)
    # get area under the curve
    roc_auc = auc(fpr, tpr)
    # PLOT ROC curve
    plt.figure(dpi=150)
    plt.plot(fpr, tpr, lw=1, color='green', label=f'AUC = {roc_auc:.3f}')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.title(title)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate (Recall)')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.legend()
    plt.show()
    # preds = probas[:, 1]
    #
    # # fpr means false-positive-rate
    # # tpr means true-positive-rate
    # fpr, tpr, _ = roc_curve(y_test, preds)
    #
    # auc_score = auc(fpr, tpr)
    #
    # # clear current figure
    # plt.clf()
    #
    # plt.title('ROC Curve')
    # plt.plot(fpr, tpr, label='AUC = {:.2f}'.format(auc_score))
    #
    # # it's helpful to add a diagonal to indicate where chance
    # # scores lie (i.e. just flipping a coin)
    # plt.plot([0, 1], [0, 1], 'r--')
    #
    # plt.xlim([-0.1, 1.1])
    # plt.ylim([-0.1, 1.1])
    # plt.ylabel('True Positive Rate')
    # plt.xlabel('False Positive Rate')
    #
    # plt.legend(loc='lower right')
    # plt.show()

def classifierMortalityUnited():

    X = df_model_draft.drop(
        ['HospID', 'SiteID', 'surgid', 'Complics', 'Mortality'], axis=1)
          # 'HospID_total_cardiac_surgery', 'HospID_Reop_CABG', 'HospID_total_CABG', 'surgyear',
          # 'surgid_total_cardiac_surgery','surgid_total_CABG', 'surgid_Reop_CABG'], axis=1)
    y = df_model_draft['Mortality']  # Labels

    print(X.shape)
    print(y.shape)
    # count examples in each class
    counter = Counter(y)
    # estimate scale_pos_weight value
    estimate = counter[0] / counter[1]
    print('Estimate: %.3f' % estimate)
    # Split dataset into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # xgb_model = xgb.XGBClassifier(objective='binary:logistic',scale_pos_weight=63.112, learning_rate=0.1, max_depth=7)  # objective="binary:logistic", random_state=42)
    xgb_model = xgb.XGBClassifier(objective='binary:logistic')
    xgb_model.fit(X_train, y_train)

    y_pred = xgb_model.predict(X_test)
    preds = xgb_model.predict_proba(X_test)
    print(confusion_matrix(y_test, y_pred))
    print(f"The accuracy of the model is {round(accuracy_score(y_test, y_pred), 5) * 100} %")

    cm = confusion_matrix(y_test, y_pred)
    labels = ['TN', 'FP', 'FN', 'TP']
    categories = ['Alive', 'Dead']
    plt = make_confusion_matrix(cm, categories=categories, cmap='RdPu',
                                title='Confusion Metrics Mortality:', group_names=labels)
    plt.show()
    feature_importance(xgb_model, df_model_draft, X_test, y_test, 'pink', 'RdPu')
    make_roc_auc_curve(y_test,preds,'ROC Curve for XGBoost with Experience')
    # print("=============================K-FOLD========================================")
    #
    # # fit xgboost on an imbalanced classification dataset
    # from numpy import mean
    # from sklearn.model_selection import cross_val_score
    # from sklearn.model_selection import RepeatedStratifiedKFold
    # model = XGBClassifier(objective='binary:logistic',scale_pos_weight=67)
    # # define evaluation procedure
    # cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
    # # evaluate model
    # scores = cross_val_score(model, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)
    # # summarize performance
    # print('Mean ROC AUC: %.5f' % mean(scores))

def classifierMortalityUnderSampling():

    X = df_train.drop(
        ['HospID', 'SiteID', 'surgid', 'Complics', 'Mortality'], axis=1)
          # 'HospID_total_cardiac_surgery', 'HospID_Reop_CABG', 'HospID_total_CABG', 'surgyear',
          # 'surgid_total_cardiac_surgery','surgid_total_CABG', 'surgid_Reop_CABG'], axis=1)
    y = df_train['Mortality']  # Labels

    X_test = df_test.drop(
        ['HospID', 'SiteID', 'surgid', 'Complics', 'Mortality'], axis=1)
    y_test = df_test['Mortality']
    # define undersample strategy
    undersample = RandomUnderSampler(sampling_strategy='majority')
    # fit and apply the transform
    X_over, y_over = undersample.fit_resample(X, y)
    # summarize class distribution
    print(Counter(y_over))
    # X_train, X_test, y_train, y_test = train_test_split(X_over, y_over, test_size=0.2)
    xgb_model = xgb.XGBClassifier(objective='binary:logistic',eval_metric= 'logloss',learning_rate=0.1)
    xgb_model.fit(X, y)

    y_pred = xgb_model.predict(X_test)
    preds = xgb_model.predict_proba(X_test)
    print(confusion_matrix(y_test, y_pred))
    print(f"The accuracy of the model is {round(accuracy_score(y_test, y_pred), 5) * 100} %")

    cm = confusion_matrix(y_test, y_pred)
    labels = ['TN', 'FP', 'FN', 'TP']
    categories = ['Alive', 'Dead']
    plt = make_confusion_matrix(cm, categories=categories, cmap='RdPu',
                                title='Confusion Metrics Mortality:', group_names=labels)
    plt.show()
    feature_importance(xgb_model, df_model_draft, X_test, y_test, 'pink', 'RdPu')
    make_roc_auc_curve(y_test,preds,'ROC Curve for XGBoost with Experience')

    # example of evaluating a decision tree with random undersampling
    from numpy import mean
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import RepeatedStratifiedKFold
    from imblearn.pipeline import Pipeline

    # define pipeline
    steps = [('under', RandomUnderSampler()), ('model', XGBClassifier())]
    pipeline = Pipeline(steps=steps)
    # evaluate pipeline
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    scores = cross_val_score(pipeline, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)
    print('Mean ROC AUC: %.5f' % mean(scores))
    print(scores)
    # scores = cross_val_score(model, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)
    # # summarize performance

# classifierMortalityUnderSampling()

def classifierMortalityNoExpereience():

    X = df_model_draft.drop(
        ['HospID', 'SiteID', 'surgid', 'Complics', 'Mortality',
          'HospID_total_cardiac_surgery', 'HospID_Reop_CABG', 'HospID_total_CABG', 'surgyear',
          'surgid_total_cardiac_surgery','surgid_total_CABG', 'surgid_Reop_CABG'], axis=1)
    y = df_model_draft['Mortality']  # Labels

    print(X.shape)
    print(y.shape)
    # count examples in each class
    counter = Counter(y)
    print (counter)
    # estimate scale_pos_weight value
    estimate = counter[0] / counter[1]
    print('Estimate: %.3f' % estimate)
    # Split dataset into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    xgb_model = xgb.XGBClassifier(objective='binary:logistic',scale_pos_weight=63.112, learning_rate=0.1, max_depth=7)  # objective="binary:logistic", random_state=42)
    xgb_model.fit(X_train, y_train)

    y_pred = xgb_model.predict(X_test)
    preds = xgb_model.predict_proba(X_test)
    print(confusion_matrix(y_test, y_pred))
    print(f"The accuracy of the model is {round(accuracy_score(y_test, y_pred), 5) * 100} %")

    cm = confusion_matrix(y_test, y_pred)
    labels = ['TN', 'FP', 'FN', 'TP']
    categories = ['Alive', 'Dead']
    plt = make_confusion_matrix(cm, categories=categories, cmap='YlGnBu',
                                title='Confusion Metrics Mortality No Experience:', group_names=labels)
    plt.show()
    feature_importance(xgb_model, df_model_draft, X_test, y_test, 'cyan', 'YlGnBu')
    make_roc_auc_curve(y_test,preds,'ROC Curve for XGBoost No Experience')
    # print("=============================K-FOLD========================================")
    #
    # # fit xgboost on an imbalanced classification dataset
    # from numpy import mean
    # from sklearn.model_selection import cross_val_score
    # from sklearn.model_selection import RepeatedStratifiedKFold
    # model = XGBClassifier(objective='binary:logistic',scale_pos_weight=67)
    # # define evaluation procedure
    # cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
    # # evaluate model
    # scores = cross_val_score(model, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)
    # # summarize performance
    # print('Mean ROC AUC: %.5f' % mean(scores))

# classifierMortalityUnderSampling()


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


def K_Fold_Split(g_search_model, model_name,  df, Y,color='YlGnBu',n=5):
    stratified_kfold = StratifiedKFold(n_splits=n, shuffle=True, random_state=4)
    metrics_score = []
    best_params_evaluation_test = []
    best_models = []
    start_KFold = timeit.default_timer()
    print("## Start {} Fold on Model '{}' ##".format(n, model_name))
    k = 1
    for train_index, test_index in stratified_kfold.split(np.zeros(len(Y)), Y):
        print("\tStart Fold Number {}".format(k))
        train_valid_X = df.iloc[train_index]
        test_X = df.iloc[test_index]

        train_valid_Y = Y.iloc[train_index]
        test_Y = Y.iloc[test_index]

        lbls_dict = {'Negative': 0,
                     'Positive': 1}

        start_nested_kfold = timeit.default_timer()
        g_search_model.fit(train_valid_X, train_valid_Y)
        stop_nested_kfold = timeit.default_timer()
        total_nested_kfold = stop_nested_kfold - start_nested_kfold
        print("\tEnd Nested KFold Process in Time : {} Sec".format(round(total_nested_kfold, 2)))
        best_params_evaluation_test.append(g_search_model.best_params_)
        best_model = g_search_model.best_estimator_
        # Evaluate(best_model, train_valid_X, train_valid_Y,lbls_dict)

        pred_Y = best_model.predict(test_X)
        cm = confusion_matrix(test_Y, pred_Y)
        metrics_test = Make_Confusion_Matrix(cm, categories=categories, cmap=color,
                                             title='SARS-Cov-2 exam result Classification on Test Set',
                                             group_names=labels)
        metrics_score.append(metrics_test)
        best_models.append(best_model)
        roc = roc_auc_score(test_Y,best_model.predict_proba(test_X))
        print ("==========================================================================================================")
        print ('K-Fold {} ROC score {}'.format(k,roc))
        print ("==========================================================================================================")

        k += 1
    stop_KFold = timeit.default_timer()
    total_KFold = stop_KFold - start_KFold
    print("Ended {} Fold Model : {}. Time : {} Sec".format(n, model_name, round(total_KFold, 2)))
    return metrics_score, best_params_evaluation_test,best_models

def Compare_K_Fold_Model_XGB(metrics_tests, best_params_rf_5Fold):
    index_rows = ['Max Depth','N Estimators','Learning Rate','F1 Score','Accuracy','Sensitivity','Specificity']
    cols = ['Fold {}'.format(i+1) for i in range(N_split)]
    XGBoost_Comperison = pd.DataFrame(columns=cols , index=index_rows)
    for i in range(N_split):
        matrics = metrics_tests[i]
        md_p = best_params_rf_5Fold[i]['xgbclassifier__max_depth']
        ne_p = best_params_rf_5Fold[i]['xgbclassifier__n_estimators']
        lr_p = best_params_rf_5Fold[i]['xgbclassifier__learning_rate']
        XGBoost_Comperison.loc[ 'Max Depth' , 'Fold {}'.format(i+1)] = md_p
        XGBoost_Comperison.loc[ 'N Estimators' , 'Fold {}'.format(i+1)] = ne_p
        XGBoost_Comperison.loc[ 'Learning Rate' , 'Fold {}'.format(i+1)] = lr_p
        for matric in matrics:
            XGBoost_Comperison.loc[ matric , 'Fold {}'.format(i+1)] = matrics[matric]
    return XGBoost_Comperison

def Create_XGBoost_Model():
    learning_rates = [0.1,0.05,0.01]
    num_estimators = [10,20,30] + list(range(45,100,5))
    max_depths = [2**x for x in range(1,7)]
    grid = {'xgbclassifier__learning_rate': learning_rates,
            'xgbclassifier__n_estimators': num_estimators,
            'xgbclassifier__max_depth': max_depths}
    xgb_model = xgb.XGBClassifier()
    cv_kfold = KFold(n_splits=N_split , shuffle = True , random_state = 4)
    pipeline = Pipeline([('under', RandomUnderSampler()), ('xgbclassifier', xgb_model)])
    xgb_model_grid_search = GridSearchCV(estimator = pipeline, param_grid = grid, cv = cv_kfold, n_jobs = -1, verbose = 4)
    return xgb_model_grid_search


def cvUnderSamplingByYears():
    X = df_train.drop(
        ['HospID', 'SiteID', 'surgid', 'Complics', 'Mortality'], axis=1)
    # 'HospID_total_cardiac_surgery', 'HospID_Reop_CABG', 'HospID_total_CABG', 'surgyear',
    # 'surgid_total_cardiac_surgery','surgid_total_CABG', 'surgid_Reop_CABG'], axis=1)
    y = df_train['Mortality']


    xgb_model_grid_search = Create_XGBoost_Model()

    metrics_tests, best_params_rf_5Fold = K_Fold_Split(xgb_model_grid_search,"XGBoost",X,y)

    comp = Compare_K_Fold_Model_XGB(metrics_tests, best_params_rf_5Fold)
    print (comp)
    comp.to_csv("best scores.csv")

# cvUnderSamplingByYears()

def trainmodelcvundersampling():
    mask = df_model_draft['surgyear'] == 2015
    df_2011 = df_model_draft[mask]
    X = df_2011.drop(
        ['HospID', 'SiteID', 'surgid', 'Complics', 'Mortality'], axis=1)
    y = df_2011['Mortality']  # Labels # Labels

    print(X.shape)
    print(y.shape)
    # Split dataset into training set and test set

    # xgb_model = xgb.XGBClassifier(
    #     objective='binary:logistic')  # ,scale_pos_weight=67.76, nthread=4)  # objective="binary:logistic", random_state=42)
    # parameters = {
    #     "learning_rate": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
    #     "max_depth": [3, 4, 5, 6, 8, 10, 12, 15],
    #     "gamma": [0.0, 0.1, 0.2, 0.3, 0.4],
    #     "colsample_bytree": [0.3, 0.4, 0.5, 0.7]
    # }
    # best_params_evaluation_test = []
    # best_models = []
    #
    # learning_rates = [0.1, 0.05, 0.01]
    # num_estimators = [10, 20, 30] + list(range(45, 100, 5))
    # max_depths = [2 ** x for x in range(1, 7)]
    # grid = {'xgbclassifier__learning_rate': learning_rates,
    #         'xgbclassifier__n_estimators': num_estimators,
    #         'xgbclassifier__max_depth': max_depths}
    # xgb_model = xgb.XGBClassifier(objective='binary:logistic')
    # cv_kfold = KFold(n_splits=N_split, shuffle=True, random_state=4)
    # cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
    # pipeline = Pipeline([('under', RandomUnderSampler()), ('xgbclassifier', xgb_model)])
    # xgb_model_grid_search = GridSearchCV(estimator=pipeline, param_grid=grid, cv=cv, n_jobs=-1, verbose=4)
    # xgb_model_grid_search.fit(X, y)
    # print (xgb_model_grid_search.best_params_)
    # print("model:")
    # print(xgb_model_grid_search.best_estimator_)
    # best_model = xgb_model_grid_search.best_estimator_
    # roc = roc_auc_score(y, best_model.predict_proba(X))
    # print("==========================================================================================================")
    # print('K-Fold  ROC score {}'.format( roc))
    # print("==========================================================================================================")
    print(Counter(y))
    undersample = RandomUnderSampler(sampling_strategy=0.35) #'majority'
    # fit and apply the transform
    X_over, y_over = undersample.fit_resample(X, y)
    # summarize class distribution
    print ("after under sampling")
    print(Counter(y_over))

    # X_train, X_test, y_train, y_test = train_test_split(X_over, y_over, test_size=0.2)
    xgb_model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', learning_rate=0.1)
    xgb_model.fit(X_over, y_over)

    X_test = df_test.drop(
        ['HospID', 'SiteID', 'surgid', 'Complics', 'Mortality'], axis=1)
    y_test = df_test['Mortality']  # Labels # Labels

    y_pred = xgb_model.predict(X_test)
    preds = xgb_model.predict_proba(X_test)
    print(confusion_matrix(y_test, y_pred))
    print(f"The accuracy of the model is {round(accuracy_score(y_test, y_pred), 5) * 100} %")

    cm = confusion_matrix(y_test, y_pred)
    print (classification_report(y_test, y_pred))
    print (roc_auc_score(y_test,preds[:, 1]))
    # labels = ['TN', 'FP', 'FN', 'TP']
    # categories = ['Alive', 'Dead']
    # plt = make_confusion_matrix(cm, categories=categories, cmap='RdPu',
    #                 title='Confusion Metrics Mortality:', group_names=labels)
    # plt.show()
    # feature_importance(xgb_model, df_model_draft, X_test, y_test, 'pink', 'RdPu')
    # make_roc_auc_curve(y_test, preds, 'ROC Curve for XGBoost with Experience')

trainmodelcvundersampling()

# trainmodelcvundersampling()

def trainmodelSmoteSampling():
    X = df_train.drop(
        ['HospID', 'SiteID', 'surgid', 'Complics', 'Mortality'], axis=1)
    y = df_train['Mortality']  # Labels # Labels

    print(X.shape)
    print(y.shape)

    undersample =SMOTE()
    # fit and apply the transform
    X_over, y_over = undersample.fit_resample(X, y)
    # summarize class distribution
    print(Counter(y_over))
    # X_train, X_test, y_train, y_test = train_test_split(X_over, y_over, test_size=0.2)
    xgb_model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', learning_rate=0.1)
    xgb_model.fit(X_over, y_over)

    X_test = df_test.drop(
        ['HospID', 'SiteID', 'surgid', 'Complics', 'Mortality'], axis=1)
    y_test = df_test['Mortality']  # Labels # Labels

    y_pred = xgb_model.predict(X_test)
    preds = xgb_model.predict_proba(X_test)
    print(confusion_matrix(y_test, y_pred))
    print(f"The accuracy of the model is {round(accuracy_score(y_test, y_pred), 5) * 100} %")

    cm = confusion_matrix(y_test, y_pred)
    labels = ['TN', 'FP', 'FN', 'TP']
    categories = ['Alive', 'Dead']
    plt = make_confusion_matrix(cm, categories=categories, cmap='RdPu',
                                title='Confusion Metrics Mortality:', group_names=labels)
    plt.show()
    feature_importance(xgb_model, df_model_draft, X_test, y_test, 'pink', 'RdPu')
    make_roc_auc_curve(y_test, preds, 'ROC Curve for XGBoost with Experience')

# trainmodelSmoteSampling()


def gridsearch():
    X = df_model_draft.drop(
        ['HospID', 'SiteID', 'surgid', 'Complics', 'Mortalty'], axis=1)
    y = df_model_draft['Mortalty']  # Labels # Labels

    print(X.shape)
    print(y.shape)
    # Split dataset into training set and test set

    xgb_model = xgb.XGBClassifier(objective='binary:logistic') #,scale_pos_weight=67.76, nthread=4)  # objective="binary:logistic", random_state=42)
    parameters =   {
    "learning_rate": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
     "max_depth": [3, 4, 5, 6, 8, 10, 12, 15],
     "gamma": [0.0, 0.1, 0.2, 0.3, 0.4],
     "colsample_bytree": [0.3, 0.4, 0.5, 0.7]
     }

    # {
    #     'max_depth': range(2, 10, 1),
    #     'n_estimators': range(60, 220, 40),
    #     'learning_rate': [0.1, 0.01, 0.05]
    # }
    # grid_search = GridSearchCV(
    #     estimator=xgb_model,
    #     param_grid=parameters,
    #     scoring='roc_auc',
    #     n_jobs=10,
    #     cv=5,
    #     verbose=True
    # )
    learning_rates = [0.1, 0.05, 0.01]
    num_estimators = [10, 20, 30] + list(range(45, 100, 5))
    max_depths = [2 ** x for x in range(1, 7)]
    grid = {'xgbclassifier__learning_rate': learning_rates,
            'xgbclassifier__n_estimators': num_estimators,
            'xgbclassifier__max_depth': max_depths}
    xgb_model = xgb.XGBClassifier(objective='binary:logistic')
    cv_kfold = KFold(n_splits=N_split, shuffle=True, random_state=4)
    pipeline = Pipeline([('under', RandomUnderSampler()), ('xgbclassifier', xgb_model)])
    xgb_model_grid_search = GridSearchCV(estimator=pipeline, param_grid=grid, cv=cv_kfold, n_jobs=-1, verbose=4)
    print(xgb_model_grid_search.fit(X, y))
    print(xgb_model_grid_search.best_estimator_)

# gridsearch()

# classifierMortalityUnited()
# classifierMortalityNoExpereience()

# def classifierComplicsUnited():
#     X = df_model_draft.drop(
#         ['HospID', 'SiteID', 'surgid', 'Complics','Mortalty',
#          'HospID_total_cardiac_surgery', 'HospID_Reop_CABG', 'HospID_total_CABG', 'surgyear',
#          'surgid_total_cardiac_surgery', 'surgid_total_CABG', 'surgid_Reop_CABG'], axis=1)
#     y = df_model_draft['Complics']  # Labels
#     # X[:50].to_csv("X BEFORE MODEL.csv")
#     print(X.shape)
#     print(y.shape)
#     # count examples in each class
#     counter = Counter(y)
#     print (counter)
#     # estimate scale_pos_weight value
#     estimate = counter[0] / counter[1]
#     print('Estimate: %.3f' % estimate)
#     # Split dataset into training set and test set
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)  # 70% training and 30% test
#     xgb_model = xgb.XGBClassifier(objective='binary:logistic',scale_pos_weight=1.65)  # objective="binary:logistic", random_state=42)
#     xgb_model.fit(X_train, y_train)
#
#     y_pred = xgb_model.predict(X_test)
#     preds = xgb_model.predict_proba(X_test)
#     print(confusion_matrix(y_test, y_pred))
#     print(f"The accuracy of the model is {round(accuracy_score(y_test, y_pred), 5) * 100} %")
#
#     cm = confusion_matrix(y_test, y_pred)
#     labels = ['TN', 'FP', 'FN', 'TP']
#     categories = ['No Complications', 'Complications']
#     plt = make_confusion_matrix(cm, categories=categories, cmap='Greens',
#                                 title='Confusion Metrics Complications:', group_names=labels)
#     plt.show()
#     feature_importance(xgb_model, df_model_draft, X_test, y_test, 'lightgreen', 'Greens')
#     make_roc_auc_curve(y_test,preds)
#
#     # print("=============================K-FOLD========================================")
#     #
#     # # fit xgboost on an imbalanced classification dataset
#     # from numpy import mean
#     # from sklearn.model_selection import cross_val_score
#     # from sklearn.model_selection import RepeatedStratifiedKFold
#     # model = XGBClassifier(objective='binary:logistic', scale_pos_weight=1.65)
#     # # define evaluation procedure
#     # cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
#     # # evaluate model
#     # scores = cross_val_score(model, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)
#     # # summarize performance
#     # print('Mean ROC AUC: %.5f' % mean(scores))

# classifierComplicsUnited()

# def binaryClassifierMortality():
#
#     X = df_model_draft.drop(
#         ['HospID', 'surgyear', 'total_surgery_count', 'total_Reop_count', 'total_CABG', 'SiteID', 'surgid', 'Complics',
#          'Mortalty'], axis=1)
#     y = df_model_draft['Mortalty']  # Labels
#     # X[:50].to_csv("X BEFORE MODEL.csv")
#     print(X.shape)
#     print(y.shape)
#     # Split dataset into training set and test set
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)  # 70% training and 30% test
#     xgb_model = xgb.XGBClassifier(objective='reg:logistic',scale_pos_weight=68) # objective="binary:logistic", random_state=42)
#     xgb_model.fit(X_train, y_train)
#
#     y_pred = xgb_model.predict(X_test)
#
#     print(confusion_matrix(y_test, y_pred))
#     print(f"The accuracy of the model is {round(accuracy_score(y_test, y_pred),5)*100} %")
#
#     cm = confusion_matrix(y_test, y_pred)
#     labels = ['TN','FP','FN','TP']
#     categories = ['Alive', 'Dead']
#     plt = make_confusion_matrix(cm,categories=categories, cmap='Greens', title='Confusion Metrics Mortality \n No xperience:', group_names=labels)
#     plt.show()
#     feature_importance(xgb_model,df_model_draft,X_test,y_test,'lightgreen','Greens')
#
#     # ====================================================================================================
#     print ("==================yearly total cardiac exp=============")
#     X = df_model_draft.drop(['HospID', 'surgyear','total_Reop_count', 'total_CABG','SiteID', 'surgid','Complics','Mortalty'], axis=1)
#     y = df_model_draft['Mortalty']  # Labels
#
#     print (X.shape)
#     print (y.shape)
#     # Split dataset into training set and test set
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
#
#     xgb_model = xgb.XGBClassifier(objective='reg:logistic') # objective="binary:logistic", random_state=42)
#     xgb_model.fit(X_train, y_train)
#
#     y_pred = xgb_model.predict(X_test)
#
#     print(confusion_matrix(y_test, y_pred))
#     print(f"The accuracy of the model is {round(accuracy_score(y_test, y_pred),5)*100} %")
#
#     cm = confusion_matrix(y_test, y_pred)
#     labels = ['TN','FP','FN','TP']
#     categories = ['Alive', 'Dead']
#     plt = make_confusion_matrix(cm,categories=categories, cmap='RdPu', title='Confusion Metrics Mortality \n Total cardiac experience:', group_names=labels)
#     plt.show()
#     feature_importance(xgb_model,df_model_draft,X_test,y_test,'pink','RdPu')
#
#     print ("")
#     # ====================================================================================================
#     print ("==================yearly total CABG  exp=============")
#     X = df_model_draft.drop(['HospID', 'surgyear', 'total_surgery_count', 'total_Reop_count','SiteID', 'surgid','Complics','Mortalty'], axis=1)
#     y = df_model_draft['Mortalty']  # Labels
#
#     print (X.shape)
#     print (y.shape)
#     # Split dataset into training set and test set
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
#
#     xgb_model = xgb.XGBClassifier(objective='reg:logistic') # objective="binary:logistic", random_state=42)
#     xgb_model.fit(X_train, y_train)
#
#     y_pred = xgb_model.predict(X_test)
#
#     print(confusion_matrix(y_test, y_pred))
#     print(f"The accuracy of the model is {round(accuracy_score(y_test, y_pred),5)*100} %")
#
#     cm = confusion_matrix(y_test, y_pred)
#     labels = ['TN','FP','FN','TP']
#     categories = ['Alive', 'Dead']
#     plt = make_confusion_matrix(cm,categories=categories, cmap='YlGnBu', title='Confusion Metrics Mortality \n Total CABG experience:', group_names=labels)
#     plt.show()
#     feature_importance(xgb_model,df_model_draft,X_test,y_test,'cyan','YlGnBu')
#
#
#     # ====================================================================================================
#     print ("")
#     print ("==================complics =============")
#     X = df_model_draft.drop(['HospID', 'surgyear', 'total_surgery_count', 'total_Reop_count', 'total_CABG','SiteID', 'surgid','Mortalty'], axis=1)
#     y = df_model_draft['Mortalty']  # Labels
#
#     print (X.shape)
#     print (y.shape)
#     # Split dataset into training set and test set
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
#
#     xgb_model = xgb.XGBClassifier(objective='reg:logistic') # objective="binary:logistic", random_state=42)
#     xgb_model.fit(X_train, y_train)
#
#     y_pred = xgb_model.predict(X_test)
#
#     print(confusion_matrix(y_test, y_pred))
#     print(f"The accuracy of the model is {round(accuracy_score(y_test, y_pred),5)*100} %")
#
#     cm = confusion_matrix(y_test, y_pred)
#     labels = ['TN','FP','FN','TP']
#     categories = ['Alive', 'Dead']
#     plt = make_confusion_matrix(cm,categories=categories, cmap='OrRd', title='Confusion Metrics Mortality \n No experience with complics:', group_names=labels)
#     plt.show()
#     feature_importance(xgb_model,df_model_draft,X_test,y_test,'bisque','OrRd')
#
#
#     print ("")
#     print ("==================complics + total experience = total CABG=============")
#
#     X = df_model_draft.drop(['HospID', 'surgyear',  'total_Reop_count','SiteID', 'surgid','Mortalty'], axis=1)
#     y = df_model_draft['Mortalty']  # Labels
#
#     print (X.shape)
#     print (y.shape)
#     # Split dataset into training set and test set
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
#
#     xgb_model = xgb.XGBClassifier(objective='reg:logistic') # objective="binary:logistic", random_state=42)
#     xgb_model.fit(X_train, y_train)
#
#     y_pred = xgb_model.predict(X_test)
#
#     print(confusion_matrix(y_test, y_pred))
#     print(f"The accuracy of the model is {round(accuracy_score(y_test, y_pred),5)*100} %")
#
#     cm = confusion_matrix(y_test, y_pred)
#     labels = ['TN','FP','FN','TP']
#     categories = ['Alive', 'Dead']
#     plt = make_confusion_matrix(cm,categories=categories, cmap='PuRd', title='Confusion Metrics Mortality \n Total cardiac and CABG experience with complics:', group_names=labels)
#     plt.show()
#     feature_importance(xgb_model,df_model_draft,X_test,y_test,'lightcoral','PuRd')
#
# def ComplicsClassifier():
#     X = df_model_draft.drop(
#         ['HospID', 'surgyear', 'total_surgery_count', 'total_Reop_count', 'total_CABG', 'SiteID', 'surgid', 'Complics',
#          'Mortalty'], axis=1)
#     y = df_model_draft['Complics']  # Labels
#     # X[:50].to_csv("X BEFORE MODEL.csv")
#     print(X.shape)
#     print(y.shape)
#     # Split dataset into training set and test set
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)  # 70% training and 30% test
#     xgb_model = xgb.XGBClassifier(objective='binary:logistic')  # objective="binary:logistic", random_state=42)
#     xgb_model.fit(X_train, y_train)
#
#     y_pred = xgb_model.predict(X_test)
#
#     print(confusion_matrix(y_test, y_pred))
#     print(f"The accuracy of the model is {round(accuracy_score(y_test, y_pred), 5) * 100} %")
#
#     cm = confusion_matrix(y_test, y_pred)
#     labels = ['TN', 'FP', 'FN', 'TP']
#     categories = ['No Complications', 'Complications']
#     plt = make_confusion_matrix(cm, categories=categories, cmap='Greens',
#                                 title='Confusion Metrics Complications \n No experience:', group_names=labels)
#     plt.show()
#     feature_importance(xgb_model, df_model_draft, X_test, y_test, 'lightgreen', 'Greens')
#
#     # ====================================================================================================
#     print("==================yearly total cardiac exp=============")
#     X = df_model_draft.drop(
#         ['HospID', 'surgyear', 'total_Reop_count', 'total_CABG', 'SiteID', 'surgid', 'Complics', 'Mortalty'], axis=1)
#     y = df_model_draft['Complics']  # Labels
#
#     print(X.shape)
#     print(y.shape)
#     # Split dataset into training set and test set
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
#
#     xgb_model = xgb.XGBClassifier(objective='binary:logistic')  # objective="binary:logistic", random_state=42)
#     xgb_model.fit(X_train, y_train)
#
#     y_pred = xgb_model.predict(X_test)
#
#     print(confusion_matrix(y_test, y_pred))
#     print(f"The accuracy of the model is {round(accuracy_score(y_test, y_pred), 5) * 100} %")
#
#     cm = confusion_matrix(y_test, y_pred)
#     labels = ['TN', 'FP', 'FN', 'TP']
#     categories = ['No Complications', 'Complications']
#     plt = make_confusion_matrix(cm, categories=categories, cmap='RdPu',
#                                 title='Confusion Metrics Complications \n Total cardiac experience:', group_names=labels)
#     plt.show()
#     feature_importance(xgb_model, df_model_draft, X_test, y_test, 'pink', 'RdPu')
#
#     print("")
#     # ====================================================================================================
#     print("==================yearly total CABG  exp=============")
#     X = df_model_draft.drop(
#         ['HospID', 'surgyear', 'total_surgery_count', 'total_Reop_count', 'SiteID', 'surgid', 'Complics', 'Mortalty'],
#         axis=1)
#     y = df_model_draft['Complics']  # Labels
#
#     print(X.shape)
#     print(y.shape)
#     # Split dataset into training set and test set
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
#
#     xgb_model = xgb.XGBClassifier(objective='binary:logistic')  # objective="binary:logistic", random_state=42)
#     xgb_model.fit(X_train, y_train)
#
#     y_pred = xgb_model.predict(X_test)
#
#     print(confusion_matrix(y_test, y_pred))
#     print(f"The accuracy of the model is {round(accuracy_score(y_test, y_pred), 5) * 100} %")
#
#     cm = confusion_matrix(y_test, y_pred)
#     labels = ['TN', 'FP', 'FN', 'TP']
#     categories = ['No Complications', 'Complications']
#     plt = make_confusion_matrix(cm, categories=categories, cmap='YlGnBu',
#                                 title='Confusion Metrics Complications \n Total CABG experience:', group_names=labels)
#     plt.show()
#     feature_importance(xgb_model, df_model_draft, X_test, y_test, 'cyan', 'YlGnBu')
#
#     # ====================================================================================================
#     print("==================yearly total CABG  exp=============")
#     X = df_model_draft.drop(
#         ['HospID', 'surgyear', 'total_Reop_count', 'SiteID', 'surgid', 'Complics', 'Mortalty'],
#         axis=1)
#     y = df_model_draft['Complics']  # Labels
#
#     print(X.shape)
#     print(y.shape)
#     # Split dataset into training set and test set
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
#
#     xgb_model = xgb.XGBClassifier(objective='binary:logistic')  # objective="binary:logistic", random_state=42)
#     xgb_model.fit(X_train, y_train)
#
#     y_pred = xgb_model.predict(X_test)
#
#     print(confusion_matrix(y_test, y_pred))
#     print(f"The accuracy of the model is {round(accuracy_score(y_test, y_pred), 5) * 100} %")
#
#     cm = confusion_matrix(y_test, y_pred)
#     labels = ['TN', 'FP', 'FN', 'TP']
#     categories = ['No Complications', 'Complications']
#     plt = make_confusion_matrix(cm, categories=categories, cmap='OrRd',
#                                 title='Confusion Metrics Complications \n Total CABG and total cardiac experience:', group_names=labels)
#     plt.show()
#     feature_importance(xgb_model, df_model_draft, X_test, y_test, 'bisque', 'OrRd')

# ComplicsClassifier()



