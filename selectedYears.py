import pandas as pd
from sklearn import linear_model
import statsmodels.api as sm
import numpy as np
from scipy import stats


df_all = pd.read_csv("/mnt/nadavrap-students/STS/data/imputed_data2.csv")



print(df_all.columns.tolist())
print (df_all.info())

df_all = df_all.replace({'MtOpD':{False:0, True:1}})
df_all = df_all.replace({'Complics':{False:0, True:1}})

mask_reop = df_all['Reoperation'] == 'Reoperation'
df_reop = df_all[mask_reop]

mask = df_all['surgyear'] == 2010
df_2010 = df_all[mask]
mask = df_all['surgyear'] == 2011
df_2011 = df_all[mask]
mask = df_all['surgyear'] == 2012
df_2012 = df_all[mask]
mask = df_all['surgyear'] == 2013
df_2013 = df_all[mask]
mask = df_all['surgyear'] == 2014
df_2014 = df_all[mask]
mask = df_all['surgyear'] == 2015
df_2015 = df_all[mask]
mask = df_all['surgyear'] == 2016
df_2016 = df_all[mask]
mask = df_all['surgyear'] == 2017
df_2017 = df_all[mask]
mask = df_all['surgyear'] == 2018
df_2018 = df_all[mask]
mask = df_all['surgyear'] == 2019
df_2019 = df_all[mask]

hospid_2019 = pd.DataFrame()

def create_2019_df(df):
    df1 = df.groupby('HospID')['HospID'].count().reset_index(name='total')
    df2 = df.groupby('HospID')['Reoperation'].apply(lambda x: (x == 'Reoperation').sum()).reset_index(name='reop')
    df3 = df.groupby('HospID')['Reoperation'].apply(lambda x: (x == 'First Time').sum()).reset_index(name='FirstOperation')
    dfmort = df.groupby('HospID')['MtOpD'].apply(lambda x: (x == 1).sum()).reset_index(name='Mortality_all')

    mask_reop = df['Reoperation'] == 'Reoperation'
    df_reop = df[mask_reop]
    df_op = df[~mask_reop]
    dfmortf = df_op.groupby('HospID')['MtOpD'].apply(lambda x: (x == 1).sum()).reset_index(name='Mortality_first')
    dfmortr = df_reop.groupby('HospID')['MtOpD'].apply(lambda x: (x == 1).sum()).reset_index(name='Mortality_reop')

    df_comp = df.groupby('HospID')['Complics'].apply(lambda x: (x == 1).sum()).reset_index(name='Complics_all')
    df_compr = df_reop.groupby('HospID')['Complics'].apply(lambda x: (x == 1).sum()).reset_index(name='Complics_reop')
    df_compf = df_op.groupby('HospID')['Complics'].apply(lambda x: (x == 1).sum()).reset_index(name='Complics_FirstOperation')

    d1 = pd.merge(df1, df3, on='HospID', how='outer')
    d2 = pd.merge(d1, df2, on='HospID', how='outer')
    d3 = pd.merge(d2, dfmort, on='HospID', how='outer')
    d4 = pd.merge(d3, dfmortf, on='HospID', how='outer')
    d5 = pd.merge(d4, dfmortr, on='HospID', how='outer')
    d6 = pd.merge(d5, df_comp, on='HospID', how='outer')
    d7 = pd.merge(d6, df_compr, on='HospID', how='outer')
    d8 = pd.merge(d7, df_compf, on='HospID', how='outer')
    #df_sum_all_Years_total = pd.merge(d8, df_19, on='HospID', how='outer')
    d8.fillna(0, inplace=True)
    d8['mort_rate_All'] = (d8['Mortality_all'] / d8['total'])*100
    d8['Mortality_First_rate'] =( d8['Mortality_first'] / d8['FirstOperation'])*100
    d8['Mortality_Reop_rate'] = (d8['Mortality_reop'] / d8['reop'])*100
    d8['Complics_rate_All'] = (d8['Complics_all'] / d8['total']) * 100
    d8['Complics_First_rate'] = (d8['Complics_FirstOperation'] / d8['FirstOperation']) * 100
    d8['Complics_Reop_rate'] = (d8['Complics_reop'] / d8['reop']) * 100
    d8.to_csv("oneyear_hospid.csv")

    df_PredMort_op = df_op.groupby('HospID')['PredMort'].mean().reset_index(name='PredMort_First_avg')
    df_PredMort_reop= df_reop.groupby('HospID')['PredMort'].mean().reset_index(name='PredMort_Reoperation_avg')

    df_PredComp_op = df_op.groupby('HospID')['PredMM'].mean().reset_index(name='PredComp_First_avg')
    df_PredComp_reop= df_reop.groupby('HospID')['PredMM'].mean().reset_index(name='PredComp_Reoperation_avg')

    d9 = pd.merge(d8, df_PredMort_op, on='HospID', how='outer')
    d10 = pd.merge(d9, df_PredMort_reop, on='HospID', how='outer')
    d11 = pd.merge(d10, df_PredComp_op, on='HospID', how='outer')
    d12 = pd.merge(d11, df_PredComp_reop, on='HospID', how='outer')
    d12.fillna(0, inplace=True)
    d12['Mort_observe/expected_First'] = (d12['Mortality_First_rate'] / d12['PredMort_First_avg'])
    d12['Mort_observe/expected_Reop'] = (d12['Mortality_Reop_rate'] / d12['PredMort_Reoperation_avg'])
    d12[['log_First_Mort', 'log_Reoperation_Mort']] = np.log2(
    d12[['Mort_observe/expected_First', 'Mort_observe/expected_Reop']].replace(0, np.nan))
    d12.fillna(0, inplace=True)

    d12['Comp_observe/expected_First'] = (d12['Complics_First_rate'] / d12['PredComp_First_avg'])
    d12['Comp_observe/expected_Reop'] = (d12['Complics_Reop_rate'] / d12['PredComp_Reoperation_avg'])
    d12[['log_First_Comp', 'log_Reoperation_Comp']] = np.log2(
    d12[['Comp_observe/expected_First', 'Comp_observe/expected_Reop']].replace(0, np.nan))
    d12.to_csv("oneyear_expec_hospid.csv")

create_2019_df(df_2017)