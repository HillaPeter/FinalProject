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



avg_hospid = pd.DataFrame()


def groupby_siteid():
    df2010 = df_2010.groupby('HospID')['HospID'].count().reset_index(name='2010_total')
    df2011 = df_2011.groupby('HospID')['HospID'].count().reset_index(name='2011_total')
    df2012 = df_2012.groupby('HospID')['HospID'].count().reset_index(name='2012_total')
    df2013 = df_2013.groupby('HospID')['HospID'].count().reset_index(name='2013_total')
    df2014 = df_2014.groupby('HospID')['HospID'].count().reset_index(name='2014_total')
    df2015 = df_2015.groupby('HospID')['HospID'].count().reset_index(name='2015_total')
    df2016 = df_2016.groupby('HospID')['HospID'].count().reset_index(name='2016_total')
    df2017 = df_2017.groupby('HospID')['HospID'].count().reset_index(name='2017_total')
    df2018 = df_2018.groupby('HospID')['HospID'].count().reset_index(name='2018_total')
    df2019 = df_2019.groupby('HospID')['HospID'].count().reset_index(name='2019_total')


    df1 =pd.merge(df2010, df2011, on='HospID', how='outer')
    df2 =pd.merge(df1, df2012, on='HospID', how='outer')
    df3 =pd.merge(df2, df2013, on='HospID', how='outer')
    df4 =pd.merge(df3, df2014, on='HospID', how='outer')
    df5 =pd.merge(df4, df2015, on='HospID', how='outer')
    df6 =pd.merge(df5, df2016, on='HospID', how='outer')
    df7 =pd.merge(df6, df2017, on='HospID', how='outer')
    df8 =pd.merge(df7, df2018, on='HospID', how='outer')
    df_sum_all_Years =pd.merge(df8, df2019, on='HospID', how='outer')

    df_sum_all_Years.fillna(0,inplace=True)

    cols = df_sum_all_Years.columns.difference(['HospID'])
    df_sum_all_Years['Distinct_years'] = df_sum_all_Years[cols].gt(0).sum(axis=1)


    cols_sum = df_sum_all_Years.columns.difference(['HospID','Distinct_years'])
    df_sum_all_Years['Year_sum'] =df_sum_all_Years.loc[:,cols_sum].sum(axis=1)
    df_sum_all_Years['Year_avg'] = df_sum_all_Years['Year_sum']/df_sum_all_Years['Distinct_years']
    df_sum_all_Years.to_csv("/tmp/pycharm_project_723/files/total op sum all years HospID.csv")
    # print("details on site id dist:")
    # # print("num of all sites: ", len(df_sum_all_Years))
    #
    # less_8 =df_sum_all_Years[df_sum_all_Years['Distinct_years'] !=10]
    # less_8.to_csv("total op less 10 years siteid.csv")
    # print("num of sites with less years: ", len(less_8))
    #
    # x = np.array(less_8['Distinct_years'])
    # print(np.unique(x))
    avg_hospid['HospID'] = df_sum_all_Years['HospID']
    avg_hospid['total_year_sum'] = df_sum_all_Years['Year_sum']
    avg_hospid['total_year_avg'] = df_sum_all_Years['Year_avg']
    avg_hospid['num_of_years'] = df_sum_all_Years['Distinct_years']

def groupby_siteid_reop():
    df2010 = df_2010.groupby('HospID')['Reoperation'].apply(lambda x: (x == 'Reoperation').sum()).reset_index(name='2010_reop')
    df2011 = df_2011.groupby('HospID')['Reoperation'].apply(lambda x: (x == 'Reoperation').sum()).reset_index(name='2011_reop')
    df2012 = df_2012.groupby('HospID')['Reoperation'].apply(lambda x: (x == 'Reoperation').sum()).reset_index(name='2012_reop')
    df2013 = df_2013.groupby('HospID')['Reoperation'].apply(lambda x: (x == 'Reoperation').sum()).reset_index(name='2013_reop')
    df2014 = df_2014.groupby('HospID')['Reoperation'].apply(lambda x: (x == 'Reoperation').sum()).reset_index(name='2014_reop')
    df2015 = df_2015.groupby('HospID')['Reoperation'].apply(lambda x: (x == 'Reoperation').sum()).reset_index(name='2015_reop')
    df2016 = df_2016.groupby('HospID')['Reoperation'].apply(lambda x: (x == 'Reoperation').sum()).reset_index(name='2016_reop')
    df2017 = df_2017.groupby('HospID')['Reoperation'].apply(lambda x: (x == 'Reoperation').sum()).reset_index(name='2017_reop')
    df2018 = df_2018.groupby('HospID')['Reoperation'].apply(lambda x: (x == 'Reoperation').sum()).reset_index(name='2018_reop')
    df2019 = df_2019.groupby('HospID')['Reoperation'].apply(lambda x: (x == 'Reoperation').sum()).reset_index(name='2019_reop')

    df1 =pd.merge(df2010, df2011, on='HospID', how='outer')
    df2 =pd.merge(df1, df2012, on='HospID', how='outer')
    df3 =pd.merge(df2, df2013, on='HospID', how='outer')
    df4 =pd.merge(df3, df2014, on='HospID', how='outer')
    df5 =pd.merge(df4, df2015, on='HospID', how='outer')
    df6 =pd.merge(df5, df2016, on='HospID', how='outer')
    df7 =pd.merge(df6, df2017, on='HospID', how='outer')
    df8 =pd.merge(df7, df2018, on='HospID', how='outer')
    df_sum_all_Years =pd.merge(df8, df2019, on='HospID', how='outer')
    df_sum_all_Years.fillna(0,inplace=True)

    cols = df_sum_all_Years.columns.difference(['HospID'])
    df_sum_all_Years['Distinct_years_reop'] = df_sum_all_Years[cols].gt(0).sum(axis=1)

    cols_sum = df_sum_all_Years.columns.difference(['HospID', 'Distinct_years_reop'])
    df_sum_all_Years['Year_sum_reop'] = df_sum_all_Years.loc[:, cols_sum].sum(axis=1)
    df_sum_all_Years['Year_avg_reop'] = df_sum_all_Years['Year_sum_reop'] / avg_hospid['num_of_years']
    df_sum_all_Years.to_csv("/tmp/pycharm_project_723/files/sum all years HospID reop.csv")

    # -----------------------first op------------------------------------

    df_10 = df_2010.groupby('HospID')['Reoperation'].apply(lambda x: (x == 'First Time').sum()).reset_index(name='2010_FirstOperation')
    df_11 = df_2011.groupby('HospID')['Reoperation'].apply(lambda x: (x == 'First Time').sum()).reset_index(name='2011_FirstOperation')
    df_12 = df_2012.groupby('HospID')['Reoperation'].apply(lambda x: (x == 'First Time').sum()).reset_index(name='2012_FirstOperation')
    df_13 = df_2013.groupby('HospID')['Reoperation'].apply(lambda x: (x == 'First Time').sum()).reset_index(name='2013_FirstOperation')
    df_14 = df_2014.groupby('HospID')['Reoperation'].apply(lambda x: (x == 'First Time').sum()).reset_index(name='2014_FirstOperation')
    df_15 = df_2015.groupby('HospID')['Reoperation'].apply(lambda x: (x == 'First Time').sum()).reset_index(name='2015_FirstOperation')
    df_16 = df_2016.groupby('HospID')['Reoperation'].apply(lambda x: (x == 'First Time').sum()).reset_index(name='2016_FirstOperation')
    df_17 = df_2017.groupby('HospID')['Reoperation'].apply(lambda x: (x == 'First Time').sum()).reset_index(name='2017_FirstOperation')
    df_18 = df_2018.groupby('HospID')['Reoperation'].apply(lambda x: (x == 'First Time').sum()).reset_index(name='2018_FirstOperation')
    df_19 = df_2019.groupby('HospID')['Reoperation'].apply(lambda x: (x == 'First Time').sum()).reset_index(name='2019_FirstOperation')

    d1 = pd.merge(df_10, df_11, on='HospID', how='outer')
    d2 = pd.merge(d1, df_12, on='HospID', how='outer')
    d3 = pd.merge(d2, df_13, on='HospID', how='outer')
    d4 = pd.merge(d3, df_14, on='HospID', how='outer')
    d5 = pd.merge(d4, df_15, on='HospID', how='outer')
    d6 = pd.merge(d5, df_16, on='HospID', how='outer')
    d7 = pd.merge(d6, df_17, on='HospID', how='outer')
    d8 = pd.merge(d7, df_18, on='HospID', how='outer')
    df_sum_all_Years_total = pd.merge(d8, df_19, on='HospID', how='outer')
    df_sum_all_Years_total.fillna(0, inplace=True)

    cols = df_sum_all_Years_total.columns.difference(['HospID'])
    df_sum_all_Years_total['Distinct_years'] = df_sum_all_Years_total[cols].gt(0).sum(axis=1)

    cols_sum = df_sum_all_Years_total.columns.difference(['HospID', 'Distinct_years'])
    df_sum_all_Years_total['Year_sum'] = df_sum_all_Years_total.loc[:, cols_sum].sum(axis=1)
    df_sum_all_Years_total['Year_avg'] = df_sum_all_Years_total['Year_sum'] / avg_hospid['num_of_years']
    df_sum_all_Years_total.to_csv("/tmp/pycharm_project_723/files/First Operation sum all years HospID.csv")

    #---------------------------merge------------------------

    temp_first = pd.DataFrame()
    temp_first['HospID'] = df_sum_all_Years_total['HospID']
    temp_first['Year_sum_Firstop'] = df_sum_all_Years_total['Year_sum']
    temp_first['Year_avg_Firstop'] = df_sum_all_Years_total['Year_avg']

    temp_reop = pd.DataFrame()
    temp_reop['HospID'] = df_sum_all_Years['HospID']
    temp_reop['Year_avg_reop'] = df_sum_all_Years['Year_avg_reop']
    temp_reop['Year_sum_reop'] = df_sum_all_Years['Year_sum_reop']

    df_mort = groupby_mortality_siteid()
    df_reop_mort = groupby_mortality_siteid_reop()
    df_reop_complics = groupby_complics_siteid()

    df20 = pd.merge(avg_hospid, temp_first, on='HospID', how='outer')
    temp_merge = pd.merge(df20, temp_reop, on='HospID', how='outer')
    temp_merge2 = pd.merge(temp_merge, df_mort, on='HospID', how='outer')
    temp_merge3 = pd.merge(temp_merge2,df_reop_mort, on='HospID', how='outer')
    total_avg_site_id = pd.merge(temp_merge3, df_reop_complics, on='HospID', how='outer')

    total_avg_site_id['firstop/total'] = (total_avg_site_id['Year_sum_Firstop'] / total_avg_site_id['total_year_sum']) *100
    total_avg_site_id['reop/total'] = (total_avg_site_id['Year_sum_reop'] / total_avg_site_id['total_year_sum']) * 100
    total_avg_site_id['mortalty_rate'] = (total_avg_site_id['Mortality'] / total_avg_site_id['total_year_sum'])*100
    total_avg_site_id['mortalty_reop_rate'] = (total_avg_site_id['Mortality_reop'] / total_avg_site_id['Year_sum_reop']) * 100
    total_avg_site_id['Complics_reop_rate'] = (total_avg_site_id['Complics_reop'] / total_avg_site_id['Year_sum_reop']) * 100
    total_avg_site_id.fillna(0, inplace=True)
    total_avg_site_id.to_csv('total_avg_HospID.csv')

    # df_siteid_reg['SiteID'] =total_avg_site_id['SiteID']
    # df_siteid_reg['total_year_avg'] = total_avg_site_id['total_year_avg']



def groupby_mortality_siteid():
    dfmort = df_all.groupby('HospID')['MtOpD'].apply(lambda x: (x == 1).sum()).reset_index(name='Mortality')
    dfmort.to_csv("/tmp/pycharm_project_723/files/mortality HospID.csv")
    return dfmort

def groupby_mortality_siteid_reop():
    dfmort = df_reop.groupby('HospID')['MtOpD'].apply(lambda x: (x == 1).sum()).reset_index(name='Mortality_reop')
    dfmort.to_csv("/tmp/pycharm_project_723/files/mortality HospID reop.csv")
    return dfmort

def groupby_complics_siteid():
    df_comp = df_all.groupby('HospID')['Complics'].apply(lambda x: (x == 1).sum()).reset_index(name='Complics')
    dfmort = df_reop.groupby('HospID')['Complics'].apply(lambda x: (x == 1).sum()).reset_index(name='Complics_reop')
    df20 = pd.merge(df_comp, dfmort, on='HospID', how='outer')
    df20.to_csv("/tmp/pycharm_project_723/files/Complics HospID.csv")
    return df20

# groupby_siteid()
# groupby_siteid_reop()

df_all = pd.read_csv("/mnt/nadavrap-students/STS/data/imputed_data2.csv")
avg_HospID = pd.read_csv("/tmp/pycharm_project_723/total_avg_HospID.csv")




df_all = df_all.replace({'MtOpD':{False:0, True:1}})
df_all = df_all.replace({'Complics':{False:0, True:1}})

mask_reop = df_all['Reoperation'] == 'Reoperation'
df_reop = df_all[mask_reop]
df_op = df_all[~mask_reop]

Mortality_siteid = pd.DataFrame()
Mortality_surgid = pd.DataFrame()
Complics_siteid = pd.DataFrame()
Complics_surgid = pd.DataFrame()

def groupby_mortality_HospID():
    df_count = df_all.groupby('HospID')['HospID'].count().reset_index(name='total')

    df_count_op = df_op.groupby('HospID')['HospID'].count().reset_index(name='count_First')
    df_mort_op = df_op.groupby('HospID')['MtOpD'].apply(lambda x: (x == 1).sum()).reset_index(name='Mortality_First')
    df_PredMort_op = df_op.groupby('HospID')['PredMort'].mean().reset_index(name='PredMort_First_avg')

    df_count_reop = df_reop.groupby('HospID')['HospID'].count().reset_index(name='count_Reop')
    df_mort_reop = df_reop.groupby('HospID')['MtOpD'].apply(lambda x: (x == 1).sum()).reset_index(name='Mortality_Reop')
    df_PredMort_reop= df_reop.groupby('HospID')['PredMort'].mean().reset_index(name='PredMort_Reoperation_avg')


    df1 = pd.merge(df_count, df_count_op, on='HospID', how='outer')
    df2 = pd.merge(df1, df_count_reop, on='HospID', how='outer')
    df3 = pd.merge(df2, df_mort_op, on='HospID', how='outer')
    df4 = pd.merge(df3, df_mort_reop, on='HospID', how='outer')
    df5 = pd.merge(df4, df_PredMort_op, on='HospID', how='outer')
    df_summarize = pd.merge(df5, df_PredMort_reop, on='HospID', how='outer')

    df_summarize.fillna(0, inplace=True)
    df_summarize['total_year_avg'] = avg_HospID['total_year_avg']
    df_summarize['Year_avg_Firstop'] = avg_HospID['Year_avg_Firstop']
    df_summarize['Year_avg_reop'] = avg_HospID['Year_avg_reop']
    df_summarize['Mortality_First_rate'] = (df_summarize['Mortality_First']/df_summarize['count_First'])*100
    df_summarize['Mortality_Reop_rate'] = (df_summarize['Mortality_Reop'] / df_summarize['count_Reop']) * 100
    df_summarize['observe/expected_First'] = (df_summarize['Mortality_First_rate']/df_summarize['PredMort_First_avg'])
    df_summarize['observe/expected_Reop'] = (df_summarize['Mortality_Reop_rate'] / df_summarize['PredMort_Reoperation_avg'])
    #df_summarize['log(observe/expected)'] = np.log10(df_summarize['observe/expected'])
    df_summarize[['log_First', 'log_Reoperation']] = np.log2(df_summarize[['observe/expected_First', 'observe/expected_Reop']].replace(0, np.nan))
    df_summarize.fillna(0, inplace=True)
    df_summarize.to_csv("mortality HospID obs vs expected.csv")
    print()
    print(df_summarize['observe/expected_First'].describe())
    print(df_summarize['observe/expected_Reop'].describe())


def groupby_complics_HospID():
    df_count = df_all.groupby('HospID')['HospID'].count().reset_index(name='total')

    df_count_op = df_op.groupby('HospID')['HospID'].count().reset_index(name='count_First')
    df_mort_op = df_op.groupby('HospID')['Complics'].apply(lambda x: (x == 1).sum()).reset_index(name='Complics_First')
    df_PredMort_op = df_op.groupby('HospID')['PredMM'].mean().reset_index(name='PredMM_First_avg')

    df_count_reop = df_reop.groupby('HospID')['HospID'].count().reset_index(name='count_Reop')
    df_mort_reop = df_reop.groupby('HospID')['Complics'].apply(lambda x: (x == 1).sum()).reset_index(name='Complics_Reop')
    df_PredMort_reop= df_reop.groupby('HospID')['PredMM'].mean().reset_index(name='PredMM_Reoperation_avg')


    df1 = pd.merge(df_count, df_count_op, on='HospID', how='outer')
    df2 = pd.merge(df1, df_count_reop, on='HospID', how='outer')
    df3 = pd.merge(df2, df_mort_op, on='HospID', how='outer')
    df4 = pd.merge(df3, df_mort_reop, on='HospID', how='outer')
    df5 = pd.merge(df4, df_PredMort_op, on='HospID', how='outer')
    df_summarize = pd.merge(df5, df_PredMort_reop, on='HospID', how='outer')

    df_summarize.fillna(0, inplace=True)
    df_summarize['total_year_avg'] = avg_HospID['total_year_avg']
    df_summarize['Year_avg_Firstop'] = avg_HospID['Year_avg_Firstop']
    df_summarize['Year_avg_reop'] = avg_HospID['Year_avg_reop']
    df_summarize['Complics_First_rate'] = (df_summarize['Complics_First']/df_summarize['count_First'])*100
    df_summarize['Complics_Reop_rate'] = (df_summarize['Complics_Reop'] / df_summarize['count_Reop']) * 100
    df_summarize['observe/expected_First'] = (df_summarize['Complics_First_rate']/df_summarize['PredMM_First_avg'])
    df_summarize['observe/expected_Reop'] = (df_summarize['Complics_Reop_rate'] / df_summarize['PredMM_Reoperation_avg'])
    #df_summarize['log(observe/expected)'] = np.log10(df_summarize['observe/expected'])
    df_summarize[['log_First', 'log_Reoperation']] = np.log2(df_summarize[['observe/expected_First', 'observe/expected_Reop']].replace(0, np.nan))
    df_summarize.fillna(0, inplace=True)
    df_summarize.to_csv("Complics HospID obs vs expected.csv")
    print()
    print("--------------------------HospID  complics-------------------------------")
    print(df_summarize['observe/expected_First'].describe())
    print(df_summarize['observe/expected_Reop'].describe())


groupby_mortality_HospID()
groupby_complics_HospID()