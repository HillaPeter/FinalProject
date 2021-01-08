import pandas as pd
from sklearn import linear_model
import statsmodels.api as sm
import numpy as np
from scipy import stats

df_all = pd.read_csv("/mnt/nadavrap-students/STS/data/imputed_data2.csv")
avg_siteid = pd.read_csv("/tmp/pycharm_project_723/total_avg_site_id.csv")
avg_surgid = pd.read_csv("/tmp/pycharm_project_723/total_avg_surgid.csv")



df_all = df_all.replace({'MtOpD':{False:0, True:1}})
df_all = df_all.replace({'Complics':{False:0, True:1}})

mask_reop = df_all['Reoperation'] == 'Reoperation'
df_reop = df_all[mask_reop]
df_op = df_all[~mask_reop]

Mortality_siteid = pd.DataFrame()
Mortality_surgid = pd.DataFrame()
Complics_siteid = pd.DataFrame()
Complics_surgid = pd.DataFrame()


def groupby_mortality_siteid():
    df_count = df_all.groupby('SiteID')['SiteID'].count().reset_index(name='total')

    df_count_op = df_op.groupby('SiteID')['SiteID'].count().reset_index(name='count_First')
    df_mort_op = df_op.groupby('SiteID')['MtOpD'].apply(lambda x: (x == 1).sum()).reset_index(name='Mortality_First')
    df_PredMort_op = df_op.groupby('SiteID')['PredMort'].mean().reset_index(name='PredMort_First_avg')

    df_count_reop = df_reop.groupby('SiteID')['SiteID'].count().reset_index(name='count_Reop')
    df_mort_reop = df_reop.groupby('SiteID')['MtOpD'].apply(lambda x: (x == 1).sum()).reset_index(name='Mortality_Reop')
    df_PredMort_reop= df_reop.groupby('SiteID')['PredMort'].mean().reset_index(name='PredMort_Reoperation_avg')


    df1 = pd.merge(df_count, df_count_op, on='SiteID', how='outer')
    df2 = pd.merge(df1, df_count_reop, on='SiteID', how='outer')
    df3 = pd.merge(df2, df_mort_op, on='SiteID', how='outer')
    df4 = pd.merge(df3, df_mort_reop, on='SiteID', how='outer')
    df5 = pd.merge(df4, df_PredMort_op, on='SiteID', how='outer')
    df_summarize = pd.merge(df5, df_PredMort_reop, on='SiteID', how='outer')

    df_summarize.fillna(0, inplace=True)
    df_summarize['total_year_avg'] = avg_siteid['total_year_avg']
    df_summarize['Year_avg_Firstop'] = avg_siteid['Year_avg_Firstop']
    df_summarize['Year_avg_reop'] = avg_siteid['Year_avg_reop']
    df_summarize['Mortality_First_rate'] = (df_summarize['Mortality_First']/df_summarize['count_First'])*100
    df_summarize['Mortality_Reop_rate'] = (df_summarize['Mortality_Reop'] / df_summarize['count_Reop']) * 100
    df_summarize['observe/expected_First'] = (df_summarize['Mortality_First_rate']/df_summarize['PredMort_First_avg'])
    df_summarize['observe/expected_Reop'] = (df_summarize['Mortality_Reop_rate'] / df_summarize['PredMort_Reoperation_avg'])
    #df_summarize['log(observe/expected)'] = np.log10(df_summarize['observe/expected'])
    df_summarize[['log_First', 'log_Reoperation']] = np.log2(df_summarize[['observe/expected_First', 'observe/expected_Reop']].replace(0, np.nan))
    df_summarize.fillna(0, inplace=True)
    df_summarize.to_csv("mortality siteid obs vs expected.csv")
    print()
    print(df_summarize['observe/expected_First'].describe())
    print(df_summarize['observe/expected_Reop'].describe())

def groupby_mortality_surgid():
    df_count = df_all.groupby('surgid')['surgid'].count().reset_index(name='total')

    df_count_op = df_op.groupby('surgid')['surgid'].count().reset_index(name='count_First')
    df_mort_op = df_op.groupby('surgid')['MtOpD'].apply(lambda x: (x == 1).sum()).reset_index(name='Mortality_First')
    df_PredMort_op = df_op.groupby('surgid')['PredMort'].mean().reset_index(name='PredMort_First_avg')

    df_count_reop = df_reop.groupby('surgid')['surgid'].count().reset_index(name='count_Reop')
    df_mort_reop = df_reop.groupby('surgid')['MtOpD'].apply(lambda x: (x == 1).sum()).reset_index(name='Mortality_Reop')
    df_PredMort_reop= df_reop.groupby('surgid')['PredMort'].mean().reset_index(name='PredMort_Reoperation_avg')


    df1 = pd.merge(df_count, df_count_op, on='surgid', how='outer')
    df2 = pd.merge(df1, df_count_reop, on='surgid', how='outer')
    df3 = pd.merge(df2, df_mort_op, on='surgid', how='outer')
    df4 = pd.merge(df3, df_mort_reop, on='surgid', how='outer')
    df5 = pd.merge(df4, df_PredMort_op, on='surgid', how='outer')
    df_summarize = pd.merge(df5, df_PredMort_reop, on='surgid', how='outer')

    df_summarize.fillna(0, inplace=True)
    df_summarize['total_year_avg'] = avg_surgid['total_year_avg']
    df_summarize['Year_avg_Firstop'] = avg_surgid['Year_avg_Firstop']
    df_summarize['Year_avg_reop'] = avg_surgid['Year_avg_reop']
    df_summarize['Mortality_First_rate'] = (df_summarize['Mortality_First']/df_summarize['count_First'])*100
    df_summarize['Mortality_Reop_rate'] = (df_summarize['Mortality_Reop'] / df_summarize['count_Reop']) * 100
    df_summarize['observe/expected_First'] = (df_summarize['Mortality_First_rate']/df_summarize['PredMort_First_avg'])
    df_summarize['observe/expected_Reop'] = (df_summarize['Mortality_Reop_rate'] / df_summarize['PredMort_Reoperation_avg'])
    #df_summarize['log(observe/expected)'] = np.log10(df_summarize['observe/expected'])
    df_summarize[['log_First', 'log_Reoperation']] = np.log2(df_summarize[['observe/expected_First', 'observe/expected_Reop']].replace(0, np.nan))
    df_summarize.fillna(0, inplace=True)
    df_summarize.to_csv("mortality surgid obs vs expected.csv")
    print("--------------------------surg id mort-------------------------------")
    print(df_summarize['Mortality_Reop_rate'].describe())
    print(df_summarize['observe/expected_Reop'].describe())

def groupby_complics_siteid():
    df_count = df_all.groupby('SiteID')['SiteID'].count().reset_index(name='total')

    df_count_op = df_op.groupby('SiteID')['SiteID'].count().reset_index(name='count_First')
    df_mort_op = df_op.groupby('SiteID')['Complics'].apply(lambda x: (x == 1).sum()).reset_index(name='Complics_First')
    df_PredMort_op = df_op.groupby('SiteID')['PredMM'].mean().reset_index(name='PredMM_First_avg')

    df_count_reop = df_reop.groupby('SiteID')['SiteID'].count().reset_index(name='count_Reop')
    df_mort_reop = df_reop.groupby('SiteID')['Complics'].apply(lambda x: (x == 1).sum()).reset_index(name='Complics_Reop')
    df_PredMort_reop= df_reop.groupby('SiteID')['PredMM'].mean().reset_index(name='PredMM_Reoperation_avg')


    df1 = pd.merge(df_count, df_count_op, on='SiteID', how='outer')
    df2 = pd.merge(df1, df_count_reop, on='SiteID', how='outer')
    df3 = pd.merge(df2, df_mort_op, on='SiteID', how='outer')
    df4 = pd.merge(df3, df_mort_reop, on='SiteID', how='outer')
    df5 = pd.merge(df4, df_PredMort_op, on='SiteID', how='outer')
    df_summarize = pd.merge(df5, df_PredMort_reop, on='SiteID', how='outer')

    df_summarize.fillna(0, inplace=True)
    df_summarize['total_year_avg'] = avg_siteid['total_year_avg']
    df_summarize['Year_avg_Firstop'] = avg_siteid['Year_avg_Firstop']
    df_summarize['Year_avg_reop'] = avg_siteid['Year_avg_reop']
    df_summarize['Complics_First_rate'] = (df_summarize['Complics_First']/df_summarize['count_First'])*100
    df_summarize['Complics_Reop_rate'] = (df_summarize['Complics_Reop'] / df_summarize['count_Reop']) * 100
    df_summarize['observe/expected_First'] = (df_summarize['Complics_First_rate']/df_summarize['PredMM_First_avg'])
    df_summarize['observe/expected_Reop'] = (df_summarize['Complics_Reop_rate'] / df_summarize['PredMM_Reoperation_avg'])
    #df_summarize['log(observe/expected)'] = np.log10(df_summarize['observe/expected'])
    df_summarize[['log_First', 'log_Reoperation']] = np.log2(df_summarize[['observe/expected_First', 'observe/expected_Reop']].replace(0, np.nan))
    df_summarize.fillna(0, inplace=True)
    df_summarize.to_csv("Complics siteid obs vs expected.csv")
    print()
    print("--------------------------site id complics-------------------------------")
    print(df_summarize['observe/expected_First'].describe())
    print(df_summarize['observe/expected_Reop'].describe())


def groupby_complics_surgeid():
    df_count = df_all.groupby('surgid')['surgid'].count().reset_index(name='total')

    df_count_op = df_op.groupby('surgid')['surgid'].count().reset_index(name='count_First')
    df_mort_op = df_op.groupby('surgid')['Complics'].apply(lambda x: (x == 1).sum()).reset_index(name='Complics_First')
    df_PredMort_op = df_op.groupby('surgid')['PredMM'].mean().reset_index(name='PredMM_First_avg')

    df_count_reop = df_reop.groupby('surgid')['surgid'].count().reset_index(name='count_Reop')
    df_mort_reop = df_reop.groupby('surgid')['Complics'].apply(lambda x: (x == 1).sum()).reset_index(
        name='Complics_Reop')
    df_PredMort_reop = df_reop.groupby('surgid')['PredMM'].mean().reset_index(name='PredMM_Reoperation_avg')

    df1 = pd.merge(df_count, df_count_op, on='surgid', how='outer')
    df2 = pd.merge(df1, df_count_reop, on='surgid', how='outer')
    df3 = pd.merge(df2, df_mort_op, on='surgid', how='outer')
    df4 = pd.merge(df3, df_mort_reop, on='surgid', how='outer')
    df5 = pd.merge(df4, df_PredMort_op, on='surgid', how='outer')
    df_summarize = pd.merge(df5, df_PredMort_reop, on='surgid', how='outer')

    df_summarize.fillna(0, inplace=True)
    df_summarize['total_year_avg'] = avg_surgid['total_year_avg']
    df_summarize['Year_avg_Firstop'] = avg_surgid['Year_avg_Firstop']
    df_summarize['Year_avg_reop'] = avg_surgid['Year_avg_reop']
    df_summarize['Complics_First_rate'] = (df_summarize['Complics_First'] / df_summarize['count_First']) * 100
    df_summarize['Complics_Reop_rate'] = (df_summarize['Complics_Reop'] / df_summarize['count_Reop']) * 100
    df_summarize['observe/expected_First'] = (df_summarize['Complics_First_rate'] / df_summarize['PredMM_First_avg'])
    df_summarize['observe/expected_Reop'] = (
                df_summarize['Complics_Reop_rate'] / df_summarize['PredMM_Reoperation_avg'])
    # df_summarize['log(observe/expected)'] = np.log10(df_summarize['observe/expected'])
    df_summarize[['log_First', 'log_Reoperation']] = np.log2(
        df_summarize[['observe/expected_First', 'observe/expected_Reop']].replace(0, np.nan))
    df_summarize.fillna(0, inplace=True)
    df_summarize.to_csv("Complics surgid obs vs expected.csv")
    print()
    print("--------------------------surg id complics-------------------------------")

    print(df_summarize['observe/expected_First'].describe())
    print(df_summarize['observe/expected_Reop'].describe())



groupby_mortality_siteid()
groupby_mortality_surgid()
groupby_complics_siteid()
groupby_complics_surgeid()


