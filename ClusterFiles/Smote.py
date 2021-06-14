import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from random import seed

from sklearn.model_selection import train_test_split, GroupShuffleSplit

from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from collections import Counter

import argparse

import shap
from sklearn.metrics import *
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.metrics import roc_auc_score, roc_curve

import warnings
warnings.filterwarnings("ignore")

labels = ['TN', 'FP', 'FN', 'TP']
categories = ['Negative', 'Positive']
N_split=5
df_all = pd.read_csv("/home/yaararum/Files/new data sum info surg and Hosp numeric values.csv")
path = "/home/yaararum/Files/model_outputs/"
# print(df_all.columns.tolist())



df_all = df_all.replace({'STSRCHOSPD':{False:0, True:1}})
df_all = df_all.replace({'Complics':{False:0, True:1}})
df_all = df_all.replace({'Mortality':{False:0, True:1}})
df_all = df_all.replace({'STSRCOM':{False:0, True:1}})
df_all = df_all.replace({'PrCVInt':{False:0, True:1}})




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

def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))

def Make_Confusion_Matrix(cf,roc=0,
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
            stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nSensitivity={:0.3f}\nF1 Score={:0.3f}\nSpecificity={:0.3f}\nAUCROC={:0.3f}".format(accuracy, precision, recall, f1_score_test , specificity,roc)
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

    # plt.show()
    CM = path+'CM '+ title + '.png'
    print(CM)
    plt.savefig(CM,bbox_inches='tight')
    return {'F1 Score': f1_score_test, 'Accuracy': accuracy, 'Sensitivity': recall, 'Specificity': specificity}


def SHAP_Iterates_SVMSMOTE( target,title,max_depth=3,n_esti=160,lr=0.1,withexperience = False, color='YlGnBu'):
    matrics = []
    seed(2145)
    groups = df_model_draft['HospID']
    if withexperience is False:
        X = df_model_draft.drop(
            ['SiteID', 'surgid', 'Complics', 'STSRCHOSPD', 'STSRCOM'], axis=1)
        y = df_model_draft[target]
    else:
        X = df_model_draft.drop(
            ['SiteID', 'surgid', 'Complics', 'STSRCHOSPD', 'STSRCOM',
             'HospID_Reop_CABG', 'HospID_total_CABG', 'surgyear', 'HospID_total_cardiac_surgery',
             'surgid_total_cardiac_surgery', 'surgid_total_CABG', 'surgid_Reop_CABG'], axis=1)
        y = df_model_draft[target]

    print(groups.shape)
    print(groups.unique())
    gss = GroupShuffleSplit(n_splits=1, train_size=.8, random_state=42)
    gss.get_n_splits()
    i = 1
    for train_idx, test_idx in gss.split(X, y, groups):
        print("TRAIN:", train_idx, "TEST:", test_idx)
        if (i == 1):
            X = X.drop(['HospID'], axis=1)

        print(X.columns.tolist())
        X_train = X.loc[train_idx]
        y_train = y.loc[train_idx]

        X_test = X.loc[test_idx]
        y_test = y.loc[test_idx]
        print("\nTRAIN DATAFRAME\n", X_train.shape)
        print("\nTEST DATAFRAME\n", X_test.shape)
        # summarize class distribution

        sm = SMOTE()  # SVMSMOTE(random_state=21)
        # fit and apply the transform
        X_over, y_over = sm.fit_resample(X_train, y_train)

        # summarize class distribution
        print("after under sampling")
        counter = Counter(y_over)
        print(counter)
        estimate = counter[0] / counter[1]
        print('Estimate: %.3f' % estimate)

        model = XGBClassifier(objective='binary:logistic', eval_metric='logloss', max_depth=max_depth, learning_rate=lr, n_estimators=n_esti)
        model.fit(X_over, y_over)
        y_pred = model.predict(X_test)

        print ("after model trained")
        explainer = shap.Explainer(model)
        shap_values = explainer(X_test,check_additivity=False)
        print ("before plot shap")
        # visualize the first prediction's explanation
        shap.plots.beeswarm(shap_values, max_display=50,show=False)
        shaptitle = path + 'SHAP ' + title + '.png'
        print(shaptitle)
        plt.savefig(shaptitle, bbox_inches='tight')

        # explainer = shap.TreeExplainer(model)
        # # Calculating the Shap values of X features
        # shap_values = explainer.shap_values(X_test,check_additivity=False)
        # shap.summary_plot(shap_values[1], X_test)
        # shaptitle = path + 'SHAP ' + title + '.png'
        # print(shaptitle)
        # plt.savefig(shaptitle, bbox_inches='tight')

        # # plt.show()
        auc = roc_auc_score(y_test, model.predict_proba(X_test.values)[:, 1])
        cm = confusion_matrix(y_test, y_pred)
        mats = Make_Confusion_Matrix(cm,roc=auc, categories=categories, cmap=color, title=title, group_names=labels,y_pred=y_pred,y_test=y_test)

        mats['AUROC'] = auc
        matrics.append(mats)
        i = i + 1
    return matrics,title


def pickle( target,title,max_depth=3,n_esti=160,lr=0.1,withexperience = False, color='YlGnBu'):
    matrics = []
    seed(2145)
    df_small = df_model_draft[df_model_draft['surgyear'].isin([2015])]
    print (df_small.shape)
    groups = df_small['HospID']
    print (groups)
    if withexperience is False:
        X = df_small.drop(
            ['SiteID', 'surgid', 'Complics', 'STSRCHOSPD', 'STSRCOM'], axis=1)
        y = df_small[target]
    else:
        X = df_small.drop(
            ['SiteID', 'surgid', 'Complics', 'STSRCHOSPD', 'STSRCOM',
             'HospID_Reop_CABG', 'HospID_total_CABG', 'surgyear', 'HospID_total_cardiac_surgery',
             'surgid_total_cardiac_surgery', 'surgid_total_CABG', 'surgid_Reop_CABG'], axis=1)
        y = df_small[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    sm = SMOTE()  # SVMSMOTE(random_state=21)
    # fit and apply the transform
    X_over, y_over = sm.fit_resample(X_train, y_train)

    # summarize class distribution
    print("after under sampling")
    counter = Counter(y_over)
    print(counter)
    estimate = counter[0] / counter[1]
    print('Estimate: %.3f' % estimate)

    model = XGBClassifier(objective='binary:logistic', eval_metric='logloss', max_depth=max_depth, learning_rate=lr, n_estimators=n_esti)
    model.fit(X_over, y_over)
    y_pred = model.predict(X_test)
    explainer = shap.Explainer(model)
    shap_values = explainer(X)

    # visualize the first prediction's explanation
    shap.plots.beeswarm(shap_values, max_display=50,show=False)
    shaptitle = path+ 'SHAP '+title+'.png'
    print(shaptitle)
    plt.savefig(shaptitle,bbox_inches='tight')
    # plt.show()
    auc = roc_auc_score(y_test, model.predict_proba(X_test.values)[:, 1])
    cm = confusion_matrix(y_test, y_pred)
    mats = Make_Confusion_Matrix(cm,roc=auc, categories=categories, cmap=color, title=title, group_names=labels,y_pred=y_pred,y_test=y_test)

    mats['AUROC'] = auc
    matrics.append(mats)

    return matrics,title


def choose_script(model):
    if model == '1':
        matrics_xgb, title = SHAP_Iterates_SVMSMOTE('STSRCOM', 'STSRCOM SMOTE without experience', 26, 100, 0.1,
                                                    True)
        matrics_xgb_df = pd.DataFrame(matrics_xgb)
        print(title)
        print(matrics_xgb_df)
    if model == '2':
        matrics_xgb, title = SHAP_Iterates_SVMSMOTE('STSRCOM', 'STSRCOM SMOTE with experience', 18, 400, 0.01)

        matrics_xgb_df = pd.DataFrame(matrics_xgb)
        print(title)
        print(matrics_xgb_df)

    if model == '3':
        matrics_xgb, title = SHAP_Iterates_SVMSMOTE('STSRCHOSPD', 'STSRCHOSPD SMOTE without experience', 18, 500,
                                                    0.01, True, color='RdPu')
        matrics_xgb_df = pd.DataFrame(matrics_xgb)
        print(title)
        print(matrics_xgb_df)

    if model == '4':
        matrics_xgb, title = SHAP_Iterates_SVMSMOTE('STSRCOM', 'STSRCOM SMOTE with experience ', 26, 100, 0.1)
        matrics_xgb_df = pd.DataFrame(matrics_xgb)
        print(title)
        print(matrics_xgb_df)


    if model == '5':
        matrics_xgb, title = SHAP_Iterates_SVMSMOTE('Complics', 'Complics SMOTE with experience', 5, 500, 0.1,
                                                    color='YlGn_r')
        matrics_xgb_df = pd.DataFrame(matrics_xgb)
        print(title)
        print(matrics_xgb_df)
    if model == '6':
        matrics_xgb, title = SHAP_Iterates_SVMSMOTE('Complics', 'Complics SMOTE without experience', 1, 100, 0.01, True,
                                                    color='YlGn_r')
        matrics_xgb_df = pd.DataFrame(matrics_xgb)
        print(title)
        print(matrics_xgb_df)
    if model == '7':
        matrics_xgb, title = SHAP_Iterates_SVMSMOTE('STSRCHOSPD', 'STSRCHOSPD SMOTE with experience', 18, 400, 0.01)

        matrics_xgb_df = pd.DataFrame(matrics_xgb)
        print(title)
        print(matrics_xgb_df)
    if model == '8':
        matrics_xgb, title = SHAP_Iterates_SVMSMOTE('STSRCHOSPD', 'STSRCHOSPD SMOTE with experience', 26, 400, 0.01,color='RdPu')

        matrics_xgb_df = pd.DataFrame(matrics_xgb)
        print(title)
        print(matrics_xgb_df)



if __name__ == "__main__":
    print("test")
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='not', type=str)
    args = parser.parse_args()
    print (args.model)
    choose_script(args.model)