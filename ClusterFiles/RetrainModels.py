import pandas as pd
import numpy as np
import os
import math
import timeit
import matplotlib.pyplot as plt
import seaborn as sns
from random import seed
from random import randint
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SVMSMOTE
from sklearn.metrics import *
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
# from sklearn.impute import IterativeImputer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.experimental import enable_iterative_imputer
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from collections import Counter

#!pip install xgboost
import xgboost as xgb
from scipy.stats import uniform, randint

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

labels = ['TN', 'FP', 'FN', 'TP']
categories = ['Negative', 'Positive']
N_split=5
df_all = pd.read_csv("/tmp/pycharm_project_723/new data sum info surg and Hosp numeric values.csv")

# print(df_all.columns.tolist())
f = open('/tmp/pycharm_project_723/model_outputs/Retrain models under sampling.txt', 'w')
print('something', file = f)


df_all = df_all.replace({'STSRCHOSPD':{False:0, True:1}})
df_all = df_all.replace({'Complics':{False:0, True:1}})
df_all = df_all.replace({'Mortality':{False:0, True:1}})
df_all = df_all.replace({'STSRCOM':{False:0, True:1}})
df_all = df_all.replace({'PrCVInt':{False:0, True:1}})

print (Counter(df_all['STSRCHOSPD']), file = f)
print (Counter(df_all['STSRCOM']), file = f)


df_all.rename(columns={"EF<=35%": "EF_less_equal_35"}, inplace=True)


df_model_draft = df_all[
        ['HospID', 'HospID_Reop_CABG', 'HospID_total_CABG', 'surgyear','HospID_total_cardiac_surgery',
         'surgid', 'surgid_total_cardiac_surgery','surgid_total_CABG', 'surgid_Reop_CABG', 'SiteID',  'Age',
         'Gender', 'RaceCaucasian', 'RaceBlack', 'RaceOther', 'Ethnicity',
         'FHCAD', 'Diabetes', 'Hypertn', 'Dyslip', 'Dialysis', 'InfEndo', 'ChrLungD', 'ImmSupp', 'PVD', 'CreatLst',
         'PrevMI', 'Arrhythmia', 'PrCVInt', 'POCPCI', 'MedACEI', 'MedASA',
         'MedBeta', 'MedInotr', 'MedNitIV', 'MedSter', 'NumDisV', 'HDEF', 'VDInsufA', 'VDStenA', 'VDInsufM', 'VDStenM',
         'VDInsufT', 'VDStenT', 'Status', 'SmokingStatus', 'InsulinDiab',
         'ModSevereLungDis', 'PreCVAorTIAorCVD', 'RenFail', 'Angina', 'UnstableAngina', 'ClassNYHGroup',
         'ArrhythAtrFibFlutter', 'ArrhythOther', 'DualAntiPlat', 'MedHeparin', 'AntiCoag',
         'MedAntiplateltNoASA', 'NumDisV_ordinal', 'LeftMain', 'EF_less_equal_35', 'BMI',
         'Complics', 'STSRCHOSPD','STSRCOM', 'Reoperation']].copy()

# df_small = df_model_draft[df_model_draft['surgyear'].isin([2015])]
# print (df_small["HospID_total_cardiac_surgery"].unique())
# df_t = df_model_draft[:1000]
# X = df_model_draft.drop(
#     ['HospID', 'SiteID', 'surgid', 'Complics', 'STSRCHOSPD','STSRCOM'], axis=1)
#      # 'HospID_Reop_CABG', 'HospID_total_CABG', 'surgyear','HospID_total_cardiac_surgery',
#      # 'surgid_total_cardiac_surgery','surgid_total_CABG', 'surgid_Reop_CABG'], axis=1)
# y = df_model_draft['STSRCOM']  # La

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



def Retrain_Model_10_Iterates_underSmapling( target,title,max_depth,n_esti,withexperience = False, color='YlGnBu'):
    matrics = []
    seed(2145)
    for i in range(10):
        rnd = randint(1, 2021)
        print("Start Iterate Number {}:".format(i+1))
        TRAIN_TEST_SPLIT_PERC = 0.8
        uniques = df_model_draft["HospID"].unique()
        sep = int(len(uniques) * TRAIN_TEST_SPLIT_PERC)
        df = df_model_draft.sample(frac=1).reset_index(drop=True)  # For shuffling your data
        train_ids, test_ids = uniques[:sep], uniques[sep:]

        train_df, test_df = df[df.HospID.isin(train_ids)], df[df.HospID.isin(test_ids)]

        print("\nTRAIN DATAFRAME\n", train_df.shape)
        print("\nTEST DATAFRAME\n", test_df.shape)
        if withexperience is False:
            X_train = train_df.drop(
                ['HospID', 'SiteID', 'surgid', 'Complics', 'STSRCHOSPD', 'STSRCOM'], axis=1)
            y_train = train_df[target]
            X_test = test_df.drop(
                ['HospID', 'SiteID', 'surgid', 'Complics', 'STSRCHOSPD', 'STSRCOM'], axis=1)
            y_test = test_df[target]
        else:
            X_train = train_df.drop(
                ['HospID', 'SiteID', 'surgid', 'Complics', 'STSRCHOSPD', 'STSRCOM',
            'HospID_Reop_CABG', 'HospID_total_CABG', 'surgyear','HospID_total_cardiac_surgery',
            'surgid_total_cardiac_surgery','surgid_total_CABG', 'surgid_Reop_CABG'], axis=1)
            y_train = train_df[target]
            X_test = test_df.drop(
                ['HospID', 'SiteID', 'surgid', 'Complics', 'STSRCHOSPD', 'STSRCOM',
                 'HospID_Reop_CABG', 'HospID_total_CABG', 'surgyear','HospID_total_cardiac_surgery',
                 'surgid_total_cardiac_surgery','surgid_total_CABG', 'surgid_Reop_CABG'], axis=1)
            y_test = test_df[target]

        undersample = RandomUnderSampler(sampling_strategy=0.33)  # 'majority'
        # fit and apply the transform
        X_over, y_over = undersample.fit_resample(X_train, y_train)

        # summarize class distribution
        print("after under sampling")
        counter = Counter(y_over)
        print(counter)
        estimate = counter[0] / counter[1]
        print('Estimate: %.3f' % estimate)

        model = XGBClassifier(objective='binary:logistic', eval_metric='logloss', max_depth=max_depth, learning_rate=0.1, n_estimators=n_esti)
                              #scale_pos_weight=estimate)
        model.fit(X_over, y_over)
        y_pred = model.predict(X_test)

        cm = confusion_matrix(y_test, y_pred)
        mats = Make_Confusion_Matrix(cm, categories=categories, cmap=color, title=title, group_names=labels,y_pred=y_pred,y_test=y_test)
        auc = roc_auc_score(y_test, model.predict_proba(X_test.values)[:, 1])
        mats['AUROC'] = auc
        matrics.append(mats)
    return matrics


# print ("yaara")
# matrics_xgb = Retrain_Model_10_Iterates_underSmapling('STSRCOM', 'STSRCOM under sampling with experience',3,180)
# matrics_xgb_df = pd.DataFrame(matrics_xgb)
# matrics_xgb_df.loc['Mean'] = matrics_xgb_df.mean()
# matrics_xgb_df.loc['Std'] = matrics_xgb_df.std()
# print(matrics_xgb_df)
# matrics_xgb_df.to_csv("/tmp/pycharm_project_723/model_outputs/STSRCOM under sampling with experience.csv")
#
# matrics_xgb = Retrain_Model_10_Iterates_underSmapling('STSRCOM', 'STSRCOM under sampling without experience',3,180,True)
# matrics_xgb_df = pd.DataFrame(matrics_xgb)
# matrics_xgb_df.loc['Mean'] = matrics_xgb_df.mean()
# matrics_xgb_df.loc['Std'] = matrics_xgb_df.std()
# print(matrics_xgb_df)
# matrics_xgb_df.to_csv("/tmp/pycharm_project_723/model_outputs/STSRCOM under sampling without experience.csv")
#
# matrics_xgb = Retrain_Model_10_Iterates_underSmapling('STSRCHOSPD', 'STSRCHOSPD under sampling with experience',3,180,color='RdPu')
# matrics_xgb_df = pd.DataFrame(matrics_xgb)
# matrics_xgb_df.loc['Mean'] = matrics_xgb_df.mean()
# matrics_xgb_df.loc['Std'] = matrics_xgb_df.std()
# print(matrics_xgb_df)
# matrics_xgb_df.to_csv("/tmp/pycharm_project_723/model_outputs/STSRCHOSPD under sampling with experience.csv")
#
# matrics_xgb = Retrain_Model_10_Iterates_underSmapling('STSRCHOSPD', 'STSRCHOSPD under sampling without experience',3,180,True,color='RdPu')
# matrics_xgb_df = pd.DataFrame(matrics_xgb)
# matrics_xgb_df.loc['Mean'] = matrics_xgb_df.mean()
# matrics_xgb_df.loc['Std'] = matrics_xgb_df.std()
# print(matrics_xgb_df)
# matrics_xgb_df.to_csv("/tmp/pycharm_project_723/model_outputs/STSRCHOSPD under sampling without experience.csv")


def Retrain_Model_10_Iterates_scalePosW( target,title,withexperience = False, color='YlGnBu'):
    matrics = []
    seed(2145)
    for i in range(10):
        rnd = randint(1, 2021)
        print("Start Iterate Number {}:".format(i+1))
        TRAIN_TEST_SPLIT_PERC = 0.8
        uniques = df_model_draft["HospID"].unique()
        sep = int(len(uniques) * TRAIN_TEST_SPLIT_PERC)
        df = df_model_draft.sample(frac=1).reset_index(drop=True)  # For shuffling your data
        train_ids, test_ids = uniques[:sep], uniques[sep:]

        train_df, test_df = df[df.HospID.isin(train_ids)], df[df.HospID.isin(test_ids)]

        print("\nTRAIN DATAFRAME\n", train_df.shape)
        print("\nTEST DATAFRAME\n", test_df.shape)
        if withexperience is False:
            X_train = train_df.drop(
                ['HospID', 'SiteID', 'surgid', 'Complics', 'STSRCHOSPD', 'STSRCOM'], axis=1)
            y_train = train_df[target]
            X_test = test_df.drop(
                ['HospID', 'SiteID', 'surgid', 'Complics', 'STSRCHOSPD', 'STSRCOM'], axis=1)
            y_test = test_df[target]
        else:
            X_train = train_df.drop(
                ['HospID', 'SiteID', 'surgid', 'Complics', 'STSRCHOSPD', 'STSRCOM',
            'HospID_Reop_CABG', 'HospID_total_CABG', 'surgyear','HospID_total_cardiac_surgery',
            'surgid_total_cardiac_surgery','surgid_total_CABG', 'surgid_Reop_CABG'], axis=1)
            y_train = train_df[target]
            X_test = test_df.drop(
                ['HospID', 'SiteID', 'surgid', 'Complics', 'STSRCHOSPD', 'STSRCOM',
                 'HospID_Reop_CABG', 'HospID_total_CABG', 'surgyear','HospID_total_cardiac_surgery',
                 'surgid_total_cardiac_surgery','surgid_total_CABG', 'surgid_Reop_CABG'], axis=1)
            y_test = test_df[target]

        undersample = RandomUnderSampler(sampling_strategy=0.33)  # 'majority'
        # fit and apply the transform
        X_over, y_over = undersample.fit_resample(X_train, y_train)

        # summarize class distribution
        print("after under sampling")
        counter = Counter(y_over)
        print(counter)
        estimate = counter[0] / counter[1]
        print('Estimate: %.3f' % estimate)

        model = XGBClassifier(objective='binary:logistic', eval_metric='logloss',scale_pos_weight=estimate)
        model.fit(X_over, y_over)
        y_pred = model.predict(X_test)

        cm = confusion_matrix(y_test, y_pred)
        mats = Make_Confusion_Matrix(cm, categories=categories, cmap=color, title=title, group_names=labels,y_pred=y_pred,y_test=y_test)
        auc = roc_auc_score(y_test, model.predict_proba(X_test.values)[:, 1])
        mats['AUROC'] = auc
        matrics.append(mats)
    return matrics

matrics_xgb = Retrain_Model_10_Iterates_scalePosW('STSRCOM', 'STSRCOM scale_pos_weight with experience')
matrics_xgb_df = pd.DataFrame(matrics_xgb)
matrics_xgb_df.loc['Mean'] = matrics_xgb_df.mean()
matrics_xgb_df.loc['Std'] = matrics_xgb_df.std()
print(matrics_xgb_df)
matrics_xgb_df.to_csv("/tmp/pycharm_project_723/model_outputs/STSRCOM scale_pos_weight with experience.csv")
#
matrics_xgb = Retrain_Model_10_Iterates_scalePosW('STSRCOM', 'STSRCOM scale_pos_weight without experience',True)
matrics_xgb_df = pd.DataFrame(matrics_xgb)
matrics_xgb_df.loc['Mean'] = matrics_xgb_df.mean()
matrics_xgb_df.loc['Std'] = matrics_xgb_df.std()
print(matrics_xgb_df)
matrics_xgb_df.to_csv("/tmp/pycharm_project_723/model_outputs/STSRCOM scale_pos_weight without experience.csv")
#
matrics_xgb = Retrain_Model_10_Iterates_scalePosW('STSRCHOSPD', 'STSRCHOSPD scale_pos_weight with experience',color='RdPu')
matrics_xgb_df = pd.DataFrame(matrics_xgb)
matrics_xgb_df.loc['Mean'] = matrics_xgb_df.mean()
matrics_xgb_df.loc['Std'] = matrics_xgb_df.std()
print(matrics_xgb_df)
matrics_xgb_df.to_csv("/tmp/pycharm_project_723/model_outputs/SSTSRCHOSPD scale_pos_weight with experience.csv")

matrics_xgb = Retrain_Model_10_Iterates_scalePosW('STSRCHOSPD', 'STSRCHOSPD scale_pos_weight without experience',True,color='RdPu')
matrics_xgb_df = pd.DataFrame(matrics_xgb)
matrics_xgb_df.loc['Mean'] = matrics_xgb_df.mean()
matrics_xgb_df.loc['Std'] = matrics_xgb_df.std()
print(matrics_xgb_df)
matrics_xgb_df.to_csv("/tmp/pycharm_project_723/model_outputs/SSTSRCHOSPD scale_pos_weight without experience.csv")

def Retrain_Model_10_Iterates_SVMSMOTE( target,title,max_depth=3,n_esti=160,withexperience = False, color='YlGnBu'):
    matrics = []
    seed(2145)
    for i in range(3):
        rnd = randint(1, 2021)
        print("Start Iterate Number {}:".format(i+1))
        TRAIN_TEST_SPLIT_PERC = 0.8
        uniques = df_model_draft["HospID"].unique()
        sep = int(len(uniques) * TRAIN_TEST_SPLIT_PERC)
        df = df_model_draft.sample(frac=1).reset_index(drop=True)  # For shuffling your data
        train_ids, test_ids = uniques[:sep], uniques[sep:]

        train_df, test_df = df[df.HospID.isin(train_ids)], df[df.HospID.isin(test_ids)]

        print("\nTRAIN DATAFRAME\n", train_df.shape)
        print("\nTEST DATAFRAME\n", test_df.shape)
        if withexperience is False:
            X_train = train_df.drop(
                ['HospID', 'SiteID', 'surgid', 'Complics', 'STSRCHOSPD', 'STSRCOM'], axis=1)
            y_train = train_df[target]
            X_test = test_df.drop(
                ['HospID', 'SiteID', 'surgid', 'Complics', 'STSRCHOSPD', 'STSRCOM'], axis=1)
            y_test = test_df[target]
        else:
            X_train = train_df.drop(
                ['HospID', 'SiteID', 'surgid', 'Complics', 'STSRCHOSPD', 'STSRCOM',
            'HospID_Reop_CABG', 'HospID_total_CABG', 'surgyear','HospID_total_cardiac_surgery',
            'surgid_total_cardiac_surgery','surgid_total_CABG', 'surgid_Reop_CABG'], axis=1)
            y_train = train_df[target]
            X_test = test_df.drop(
                ['HospID', 'SiteID', 'surgid', 'Complics', 'STSRCHOSPD', 'STSRCOM',
                 'HospID_Reop_CABG', 'HospID_total_CABG', 'surgyear','HospID_total_cardiac_surgery',
                 'surgid_total_cardiac_surgery','surgid_total_CABG', 'surgid_Reop_CABG'], axis=1)
            y_test = test_df[target]

        sm = SVMSMOTE()  # SVMSMOTE(random_state=21)
        # fit and apply the transform
        X_over, y_over = sm.fit_resample(X_train, y_train)

        # summarize class distribution
        print("after under sampling")
        counter = Counter(y_over)
        print(counter)
        estimate = counter[0] / counter[1]
        print('Estimate: %.3f' % estimate)

        model = XGBClassifier(objective='binary:logistic', eval_metric='logloss', max_depth=max_depth, learning_rate=0.1, n_estimators=n_esti)
        model.fit(X_over, y_over)
        y_pred = model.predict(X_test)

        cm = confusion_matrix(y_test, y_pred)
        mats = Make_Confusion_Matrix(cm, categories=categories, cmap=color, title=title, group_names=labels,y_pred=y_pred,y_test=y_test)
        auc = roc_auc_score(y_test, model.predict_proba(X_test.values)[:, 1])
        mats['AUROC'] = auc
        matrics.append(mats)
    return matrics

# matrics_xgb = Retrain_Model_10_Iterates_SVMSMOTE('STSRCOM', 'STSRCOM SVMSMOTE with experience',3,180)
# matrics_xgb_df = pd.DataFrame(matrics_xgb)
# matrics_xgb_df.loc['Mean'] = matrics_xgb_df.mean()
# matrics_xgb_df.loc['Std'] = matrics_xgb_df.std()
# print(matrics_xgb_df)
# matrics_xgb_df.to_csv("/model_outputs/STSRCOM SVMSMOTE with experience.csv")
#
# matrics_xgb = Retrain_Model_10_Iterates_SVMSMOTE('STSRCOM', 'STSRCOM SVMSMOTE without experience',3,180,True)
# matrics_xgb_df = pd.DataFrame(matrics_xgb)
# matrics_xgb_df.loc['Mean'] = matrics_xgb_df.mean()
# matrics_xgb_df.loc['Std'] = matrics_xgb_df.std()
# print(matrics_xgb_df)
# matrics_xgb_df.to_csv("/model_outputs/STSRCOM SVMSMOTE without experience.csv")
#
# matrics_xgb = Retrain_Model_10_Iterates_SVMSMOTE('STSRCHOSPD', 'STSRCHOSPD SVMSMOTE with experience',3,180,color='RdPu')
# matrics_xgb_df = pd.DataFrame(matrics_xgb)
# matrics_xgb_df.loc['Mean'] = matrics_xgb_df.mean()
# matrics_xgb_df.loc['Std'] = matrics_xgb_df.std()
# print(matrics_xgb_df)
# matrics_xgb_df.to_csv("/model_outputs/STSRCHOSPD SVMSMOTE with experience.csv")
#
# matrics_xgb = Retrain_Model_10_Iterates_SVMSMOTE('STSRCHOSPD', 'STSRCHOSPD SVMSMOTE without experience',3,180,True,color='RdPu')
# matrics_xgb_df = pd.DataFrame(matrics_xgb)
# matrics_xgb_df.loc['Mean'] = matrics_xgb_df.mean()
# matrics_xgb_df.loc['Std'] = matrics_xgb_df.std()
# print(matrics_xgb_df)
# matrics_xgb_df.to_csv("/model_outputs/STSRCHOSPD SVMSMOTE without experience.csv")