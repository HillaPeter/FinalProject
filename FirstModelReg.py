import pandas as pd
from sklearn import linear_model
import statsmodels.api as sm
import numpy as np
from scipy import stats


df_all = pd.read_csv("/mnt/nadavrap-students/STS/data/imputed_data2.csv")

# print(df_all.head())
#
# print(df_all.columns.tolist())

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



avg_siteid = pd.DataFrame()
avg_surgid = pd.DataFrame()
df_siteid_reg = pd.DataFrame()
df_surgid_reg = pd.DataFrame()

def groupby_siteid():
    df2010 = df_2010.groupby('SiteID')['SiteID'].count().reset_index(name='2010_total')
    df2011 = df_2011.groupby('SiteID')['SiteID'].count().reset_index(name='2011_total')
    df2012 = df_2012.groupby('SiteID')['SiteID'].count().reset_index(name='2012_total')
    df2013 = df_2013.groupby('SiteID')['SiteID'].count().reset_index(name='2013_total')
    df2014 = df_2014.groupby('SiteID')['SiteID'].count().reset_index(name='2014_total')
    df2015 = df_2015.groupby('SiteID')['SiteID'].count().reset_index(name='2015_total')
    df2016 = df_2016.groupby('SiteID')['SiteID'].count().reset_index(name='2016_total')
    df2017 = df_2017.groupby('SiteID')['SiteID'].count().reset_index(name='2017_total')
    df2018 = df_2018.groupby('SiteID')['SiteID'].count().reset_index(name='2018_total')
    df2019 = df_2019.groupby('SiteID')['SiteID'].count().reset_index(name='2019_total')


    df1 =pd.merge(df2010, df2011, on='SiteID', how='outer')
    df2 =pd.merge(df1, df2012, on='SiteID', how='outer')
    df3 =pd.merge(df2, df2013, on='SiteID', how='outer')
    df4 =pd.merge(df3, df2014, on='SiteID', how='outer')
    df5 =pd.merge(df4, df2015, on='SiteID', how='outer')
    df6 =pd.merge(df5, df2016, on='SiteID', how='outer')
    df7 =pd.merge(df6, df2017, on='SiteID', how='outer')
    df8 =pd.merge(df7, df2018, on='SiteID', how='outer')
    df_sum_all_Years =pd.merge(df8, df2019, on='SiteID', how='outer')

    df_sum_all_Years.fillna(0,inplace=True)

    cols = df_sum_all_Years.columns.difference(['SiteID'])
    df_sum_all_Years['Distinct_years'] = df_sum_all_Years[cols].gt(0).sum(axis=1)


    cols_sum = df_sum_all_Years.columns.difference(['SiteID','Distinct_years'])
    df_sum_all_Years['Year_sum'] =df_sum_all_Years.loc[:,cols_sum].sum(axis=1)
    df_sum_all_Years['Year_avg'] = df_sum_all_Years['Year_sum']/df_sum_all_Years['Distinct_years']
    df_sum_all_Years.to_csv("/tmp/pycharm_project_723/files/total op sum all years siteid.csv")
    # print("details on site id dist:")
    # # print("num of all sites: ", len(df_sum_all_Years))
    #
    # less_8 =df_sum_all_Years[df_sum_all_Years['Distinct_years'] !=10]
    # less_8.to_csv("total op less 10 years siteid.csv")
    # print("num of sites with less years: ", len(less_8))
    #
    # x = np.array(less_8['Distinct_years'])
    # print(np.unique(x))
    avg_siteid['SiteID'] = df_sum_all_Years['SiteID']
    avg_siteid['total_year_sum'] = df_sum_all_Years['Year_sum']
    avg_siteid['total_year_avg'] = df_sum_all_Years['Year_avg']
    avg_siteid['num_of_years'] = df_sum_all_Years['Distinct_years']

def groupby_siteid_reop():
    df2010 = df_2010.groupby('SiteID')['Reoperation'].apply(lambda x: (x == 'Reoperation').sum()).reset_index(name='2010_reop')
    df2011 = df_2011.groupby('SiteID')['Reoperation'].apply(lambda x: (x == 'Reoperation').sum()).reset_index(name='2011_reop')
    df2012 = df_2012.groupby('SiteID')['Reoperation'].apply(lambda x: (x == 'Reoperation').sum()).reset_index(name='2012_reop')
    df2013 = df_2013.groupby('SiteID')['Reoperation'].apply(lambda x: (x == 'Reoperation').sum()).reset_index(name='2013_reop')
    df2014 = df_2014.groupby('SiteID')['Reoperation'].apply(lambda x: (x == 'Reoperation').sum()).reset_index(name='2014_reop')
    df2015 = df_2015.groupby('SiteID')['Reoperation'].apply(lambda x: (x == 'Reoperation').sum()).reset_index(name='2015_reop')
    df2016 = df_2016.groupby('SiteID')['Reoperation'].apply(lambda x: (x == 'Reoperation').sum()).reset_index(name='2016_reop')
    df2017 = df_2017.groupby('SiteID')['Reoperation'].apply(lambda x: (x == 'Reoperation').sum()).reset_index(name='2017_reop')
    df2018 = df_2018.groupby('SiteID')['Reoperation'].apply(lambda x: (x == 'Reoperation').sum()).reset_index(name='2018_reop')
    df2019 = df_2019.groupby('SiteID')['Reoperation'].apply(lambda x: (x == 'Reoperation').sum()).reset_index(name='2019_reop')

    df1 =pd.merge(df2010, df2011, on='SiteID', how='outer')
    df2 =pd.merge(df1, df2012, on='SiteID', how='outer')
    df3 =pd.merge(df2, df2013, on='SiteID', how='outer')
    df4 =pd.merge(df3, df2014, on='SiteID', how='outer')
    df5 =pd.merge(df4, df2015, on='SiteID', how='outer')
    df6 =pd.merge(df5, df2016, on='SiteID', how='outer')
    df7 =pd.merge(df6, df2017, on='SiteID', how='outer')
    df8 =pd.merge(df7, df2018, on='SiteID', how='outer')
    df_sum_all_Years =pd.merge(df8, df2019, on='SiteID', how='outer')
    df_sum_all_Years.fillna(0,inplace=True)

    cols = df_sum_all_Years.columns.difference(['SiteID'])
    df_sum_all_Years['Distinct_years_reop'] = df_sum_all_Years[cols].gt(0).sum(axis=1)

    cols_sum = df_sum_all_Years.columns.difference(['SiteID', 'Distinct_years_reop'])
    df_sum_all_Years['Year_sum_reop'] = df_sum_all_Years.loc[:, cols_sum].sum(axis=1)
    df_sum_all_Years['Year_avg_reop'] = df_sum_all_Years['Year_sum_reop'] / avg_siteid['num_of_years']
    df_sum_all_Years.to_csv("/tmp/pycharm_project_723/files/sum all years siteid reop.csv")

    # -----------------------first op------------------------------------

    df_10 = df_2010.groupby('SiteID')['Reoperation'].apply(lambda x: (x == 'First Time').sum()).reset_index(name='2010_FirstOperation')
    df_11 = df_2011.groupby('SiteID')['Reoperation'].apply(lambda x: (x == 'First Time').sum()).reset_index(name='2011_FirstOperation')
    df_12 = df_2012.groupby('SiteID')['Reoperation'].apply(lambda x: (x == 'First Time').sum()).reset_index(name='2012_FirstOperation')
    df_13 = df_2013.groupby('SiteID')['Reoperation'].apply(lambda x: (x == 'First Time').sum()).reset_index(name='2013_FirstOperation')
    df_14 = df_2014.groupby('SiteID')['Reoperation'].apply(lambda x: (x == 'First Time').sum()).reset_index(name='2014_FirstOperation')
    df_15 = df_2015.groupby('SiteID')['Reoperation'].apply(lambda x: (x == 'First Time').sum()).reset_index(name='2015_FirstOperation')
    df_16 = df_2016.groupby('SiteID')['Reoperation'].apply(lambda x: (x == 'First Time').sum()).reset_index(name='2016_FirstOperation')
    df_17 = df_2017.groupby('SiteID')['Reoperation'].apply(lambda x: (x == 'First Time').sum()).reset_index(name='2017_FirstOperation')
    df_18 = df_2018.groupby('SiteID')['Reoperation'].apply(lambda x: (x == 'First Time').sum()).reset_index(name='2018_FirstOperation')
    df_19 = df_2019.groupby('SiteID')['Reoperation'].apply(lambda x: (x == 'First Time').sum()).reset_index(name='2019_FirstOperation')

    d1 = pd.merge(df_10, df_11, on='SiteID', how='outer')
    d2 = pd.merge(d1, df_12, on='SiteID', how='outer')
    d3 = pd.merge(d2, df_13, on='SiteID', how='outer')
    d4 = pd.merge(d3, df_14, on='SiteID', how='outer')
    d5 = pd.merge(d4, df_15, on='SiteID', how='outer')
    d6 = pd.merge(d5, df_16, on='SiteID', how='outer')
    d7 = pd.merge(d6, df_17, on='SiteID', how='outer')
    d8 = pd.merge(d7, df_18, on='SiteID', how='outer')
    df_sum_all_Years_total = pd.merge(d8, df_19, on='SiteID', how='outer')
    df_sum_all_Years_total.fillna(0, inplace=True)

    cols = df_sum_all_Years_total.columns.difference(['SiteID'])
    df_sum_all_Years_total['Distinct_years'] = df_sum_all_Years_total[cols].gt(0).sum(axis=1)

    cols_sum = df_sum_all_Years_total.columns.difference(['SiteID', 'Distinct_years'])
    df_sum_all_Years_total['Year_sum'] = df_sum_all_Years_total.loc[:, cols_sum].sum(axis=1)
    df_sum_all_Years_total['Year_avg'] = df_sum_all_Years_total['Year_sum'] / avg_siteid['num_of_years']
    df_sum_all_Years_total.to_csv("/tmp/pycharm_project_723/files/First Operation sum all years siteid.csv")

    #---------------------------merge------------------------

    temp_first = pd.DataFrame()
    temp_first['SiteID'] = df_sum_all_Years_total['SiteID']
    temp_first['Year_sum_Firstop'] = df_sum_all_Years_total['Year_sum']
    temp_first['Year_avg_Firstop'] = df_sum_all_Years_total['Year_avg']

    temp_reop = pd.DataFrame()
    temp_reop['SiteID'] = df_sum_all_Years['SiteID']
    temp_reop['Year_avg_reop'] = df_sum_all_Years['Year_avg_reop']
    temp_reop['Year_sum_reop'] = df_sum_all_Years['Year_sum_reop']

    df_mort = groupby_mortality_siteid()
    df_reop_mort = groupby_mortality_siteid_reop()
    df_reop_complics = groupby_complics_siteid()

    df20 = pd.merge(avg_siteid, temp_first, on='SiteID', how='outer')
    temp_merge = pd.merge(df20, temp_reop, on='SiteID', how='outer')
    temp_merge2 = pd.merge(temp_merge, df_mort, on='SiteID', how='outer')
    temp_merge3 = pd.merge(temp_merge2,df_reop_mort, on='SiteID', how='outer')
    total_avg_site_id = pd.merge(temp_merge3, df_reop_complics, on='SiteID', how='outer')

    total_avg_site_id['firstop/total'] = (total_avg_site_id['Year_sum_Firstop'] / total_avg_site_id['total_year_sum']) *100
    total_avg_site_id['reop/total'] = (total_avg_site_id['Year_sum_reop'] / total_avg_site_id['total_year_sum']) * 100
    total_avg_site_id['mortalty_rate'] = (total_avg_site_id['Mortality'] / total_avg_site_id['total_year_sum'])*100
    total_avg_site_id['mortalty_reop_rate'] = (total_avg_site_id['Mortality_reop'] / total_avg_site_id['Year_sum_reop']) * 100
    total_avg_site_id['Complics_reop_rate'] = (total_avg_site_id['Complics_reop'] / total_avg_site_id['Year_sum_reop']) * 100
    total_avg_site_id.fillna(0, inplace=True)
    total_avg_site_id.to_csv('total_avg_site_id.csv')

    df_siteid_reg['SiteID'] =total_avg_site_id['SiteID']
    df_siteid_reg['total_year_avg'] = total_avg_site_id['total_year_avg']

def groupby_surgid():
    df2010 = df_2010.groupby('surgid')['surgid'].count().reset_index(name='2010_total')
    df2011 = df_2011.groupby('surgid')['surgid'].count().reset_index(name='2011_total')
    df2012 = df_2012.groupby('surgid')['surgid'].count().reset_index(name='2012_total')
    df2013 = df_2013.groupby('surgid')['surgid'].count().reset_index(name='2013_total')
    df2014 = df_2014.groupby('surgid')['surgid'].count().reset_index(name='2014_total')
    df2015 = df_2015.groupby('surgid')['surgid'].count().reset_index(name='2015_total')
    df2016 = df_2016.groupby('surgid')['surgid'].count().reset_index(name='2016_total')
    df2017 = df_2017.groupby('surgid')['surgid'].count().reset_index(name='2017_total')
    df2018 = df_2018.groupby('surgid')['surgid'].count().reset_index(name='2018_total')
    df2019 = df_2019.groupby('surgid')['surgid'].count().reset_index(name='2019_total')

    df1 = pd.merge(df2010, df2011, on='surgid', how='outer')
    df2 = pd.merge(df1, df2012, on='surgid', how='outer')
    df3 = pd.merge(df2, df2013, on='surgid', how='outer')
    df4 = pd.merge(df3, df2014, on='surgid', how='outer')
    df5 = pd.merge(df4, df2015, on='surgid', how='outer')
    df6 = pd.merge(df5, df2016, on='surgid', how='outer')
    df7 = pd.merge(df6, df2017, on='surgid', how='outer')
    df8 = pd.merge(df7, df2018, on='surgid', how='outer')
    df_sum_all_Years = pd.merge(df8, df2019, on='surgid', how='outer')

    df_sum_all_Years.fillna(0, inplace=True)

    cols = df_sum_all_Years.columns.difference(['surgid'])
    df_sum_all_Years['Distinct_years'] = df_sum_all_Years[cols].gt(0).sum(axis=1)

    cols_sum = df_sum_all_Years.columns.difference(['surgid', 'Distinct_years'])
    df_sum_all_Years['Year_sum'] = df_sum_all_Years.loc[:, cols_sum].sum(axis=1)
    df_sum_all_Years['Year_avg'] = df_sum_all_Years['Year_sum'] / df_sum_all_Years['Distinct_years']
    df_sum_all_Years.to_csv("/tmp/pycharm_project_723/files/total op sum all years surgid.csv")
    # print("details on surg id dist:")
    # print("num of all surgid: ", len(df_sum_all_Years))
    #
    # less_8 = df_sum_all_Years[df_sum_all_Years['Distinct_years'] != 10]
    # less_8.to_csv("total op less 10 years surgid.csv")
    # print("num of surgid with less years: ", len(less_8))
    #
    # x = np.array(less_8['Distinct_years'])
    # print(np.unique(x))
    # avg_surgid['surgid'] = df_sum_all_Years['surgid']
    # avg_surgid['total_year_avg'] = df_sum_all_Years['Year_avg']
    avg_surgid['surgid'] = df_sum_all_Years['surgid']
    avg_surgid['total_year_avg'] = df_sum_all_Years['Year_avg']
    avg_surgid['total_year_count'] = df_sum_all_Years['Year_sum']
    avg_surgid['num_of_years'] = df_sum_all_Years['Distinct_years']

def groupby_surgid_reop():
    df2010 = df_2010.groupby('surgid')['Reoperation'].apply(lambda x: (x =='Reoperation').sum()).reset_index(name='2010_reop')
    df2011 = df_2011.groupby('surgid')['Reoperation'].apply(lambda x: (x =='Reoperation').sum()).reset_index(name='2011_reop')
    df2012 = df_2012.groupby('surgid')['Reoperation'].apply(lambda x: (x =='Reoperation').sum()).reset_index(name='2012_reop')
    df2013 = df_2013.groupby('surgid')['Reoperation'].apply(lambda x: (x =='Reoperation').sum()).reset_index(name='2013_reop')
    df2014 = df_2014.groupby('surgid')['Reoperation'].apply(lambda x: (x =='Reoperation').sum()).reset_index(name='2014_reop')
    df2015 = df_2015.groupby('surgid')['Reoperation'].apply(lambda x: (x =='Reoperation').sum()).reset_index(name='2015_reop')
    df2016 = df_2016.groupby('surgid')['Reoperation'].apply(lambda x: (x =='Reoperation').sum()).reset_index(name='2016_reop')
    df2017 = df_2017.groupby('surgid')['Reoperation'].apply(lambda x: (x =='Reoperation').sum()).reset_index(name='2017_reop')
    df2018 = df_2018.groupby('surgid')['Reoperation'].apply(lambda x: (x =='Reoperation').sum()).reset_index(name='2018_reop')
    df2019 = df_2019.groupby('surgid')['Reoperation'].apply(lambda x: (x =='Reoperation').sum()).reset_index(name='2019_reop')

    df1 = pd.merge(df2010, df2011, on='surgid', how='outer')
    df2 = pd.merge(df1, df2012, on='surgid', how='outer')
    df3 = pd.merge(df2, df2013, on='surgid', how='outer')
    df4 = pd.merge(df3, df2014, on='surgid', how='outer')
    df5 = pd.merge(df4, df2015, on='surgid', how='outer')
    df6 = pd.merge(df5, df2016, on='surgid', how='outer')
    df7 = pd.merge(df6, df2017, on='surgid', how='outer')
    df8 = pd.merge(df7, df2018, on='surgid', how='outer')
    df_sum_all_Years = pd.merge(df8, df2019, on='surgid', how='outer')
    df_sum_all_Years.fillna(0, inplace=True)

    cols = df_sum_all_Years.columns.difference(['surgid'])
    df_sum_all_Years['Distinct_years_reop'] = df_sum_all_Years[cols].gt(0).sum(axis=1)

    cols_sum = df_sum_all_Years.columns.difference(['surgid', 'Distinct_years_reop'])
    df_sum_all_Years['Year_sum_reop'] = df_sum_all_Years.loc[:, cols_sum].sum(axis=1)
    df_sum_all_Years['Year_avg_reop'] = df_sum_all_Years['Year_sum_reop'] / avg_surgid['num_of_years']
    df_sum_all_Years.to_csv("/tmp/pycharm_project_723/files/sum all years surgid reop.csv")

    # -----------------------first op------------------------------------
    df_10 = df_2010.groupby('surgid')['Reoperation'].apply(lambda x: (x == 'First Time').sum()).reset_index(name='2010_FirstOperation')
    df_11 = df_2011.groupby('surgid')['Reoperation'].apply(lambda x: (x == 'First Time').sum()).reset_index(name='2011_FirstOperation')
    df_12 = df_2012.groupby('surgid')['Reoperation'].apply(lambda x: (x == 'First Time').sum()).reset_index(name='2012_FirstOperation')
    df_13 = df_2013.groupby('surgid')['Reoperation'].apply(lambda x: (x == 'First Time').sum()).reset_index(name='2013_FirstOperation')
    df_14 = df_2014.groupby('surgid')['Reoperation'].apply(lambda x: (x == 'First Time').sum()).reset_index(name='2014_FirstOperation')
    df_15 = df_2015.groupby('surgid')['Reoperation'].apply(lambda x: (x == 'First Time').sum()).reset_index(name='2015_FirstOperation')
    df_16 = df_2016.groupby('surgid')['Reoperation'].apply(lambda x: (x == 'First Time').sum()).reset_index(name='2016_FirstOperation')
    df_17 = df_2017.groupby('surgid')['Reoperation'].apply(lambda x: (x == 'First Time').sum()).reset_index(name='2017_FirstOperation')
    df_18 = df_2018.groupby('surgid')['Reoperation'].apply(lambda x: (x == 'First Time').sum()).reset_index(name='2018_FirstOperation')
    df_19 = df_2019.groupby('surgid')['Reoperation'].apply(lambda x: (x == 'First Time').sum()).reset_index(name='2019_FirstOperation')

    d1 = pd.merge(df_10, df_11, on='surgid', how='outer')
    d2 = pd.merge(d1, df_12, on='surgid', how='outer')
    d3 = pd.merge(d2, df_13, on='surgid', how='outer')
    d4 = pd.merge(d3, df_14, on='surgid', how='outer')
    d5 = pd.merge(d4, df_15, on='surgid', how='outer')
    d6 = pd.merge(d5, df_16, on='surgid', how='outer')
    d7 = pd.merge(d6, df_17, on='surgid', how='outer')
    d8 = pd.merge(d7, df_18, on='surgid', how='outer')
    df_sum_all_Years_total = pd.merge(d8, df_19, on='surgid', how='outer')
    df_sum_all_Years_total.fillna(0, inplace=True)

    cols = df_sum_all_Years_total.columns.difference(['surgid'])
    df_sum_all_Years_total['Distinct_years'] = df_sum_all_Years_total[cols].gt(0).sum(axis=1)

    cols_sum = df_sum_all_Years_total.columns.difference(['surgid', 'Distinct_years'])
    df_sum_all_Years_total['Year_sum'] = df_sum_all_Years_total.loc[:, cols_sum].sum(axis=1)
    df_sum_all_Years_total['Year_avg'] = df_sum_all_Years_total['Year_sum'] / avg_surgid['num_of_years']
    df_sum_all_Years_total.to_csv("/tmp/pycharm_project_723/files/First op sum all years surgid.csv")

    # ---------------------------merge------------------------

    temp_first = pd.DataFrame()
    temp_first['surgid'] = df_sum_all_Years_total['surgid']
    temp_first['Year_avg_Firstop'] = df_sum_all_Years_total['Year_avg']
    temp_first['Year_sum_Firstop'] = df_sum_all_Years_total['Year_sum']
    temp_reop = pd.DataFrame()
    temp_reop['surgid'] = df_sum_all_Years['surgid']
    temp_reop['Year_avg_reop'] = df_sum_all_Years['Year_avg_reop']
    temp_reop['Year_sum_reop'] = df_sum_all_Years['Year_sum_reop']


    df_mort = groupby_mortality_surgid()
    df_reop_mort = groupby_mortality_surgid_reop()
    df_reop_complics = groupby_complics_surgid()

    df20 = pd.merge(avg_surgid, temp_first, on='surgid', how='outer')
    temp_merge = pd.merge(df20, temp_reop, on='surgid', how='outer')
    temp_merge2 = pd.merge(temp_merge, df_mort, on='surgid', how='outer')
    temp_merge3 = pd.merge(temp_merge2,df_reop_mort, on='surgid', how='outer')
    total_avg_surgid = pd.merge(temp_merge3, df_reop_complics, on='surgid', how='outer')


    total_avg_surgid['firstop/total'] = (total_avg_surgid['Year_sum_Firstop'] / total_avg_surgid['total_year_count']) * 100
    total_avg_surgid['reop/total'] = (total_avg_surgid['Year_sum_reop'] / total_avg_surgid['total_year_count']) * 100
    total_avg_surgid['mortalty_rate'] = (total_avg_surgid['Mortality'] / total_avg_surgid['total_year_count']) * 100
    total_avg_surgid['mortalty_reop_rate'] = (total_avg_surgid['Mortality_reop'] / total_avg_surgid['Year_sum_reop']) * 100
    total_avg_surgid['Complics_reop_rate'] = (total_avg_surgid['Complics_reop'] / total_avg_surgid['Year_sum_reop']) * 100
    total_avg_surgid.fillna(0, inplace=True)
    total_avg_surgid.to_csv('total_avg_surgid.csv')

def groupby_mortality_siteid():
    dfmort = df_all.groupby('SiteID')['MtOpD'].apply(lambda x: (x == 1).sum()).reset_index(name='Mortality')
    dfmort.to_csv("/tmp/pycharm_project_723/files/mortality siteid.csv")
    return dfmort

def groupby_mortality_siteid_reop():
    dfmort = df_reop.groupby('SiteID')['MtOpD'].apply(lambda x: (x == 1).sum()).reset_index(name='Mortality_reop')
    dfmort.to_csv("/tmp/pycharm_project_723/files/mortality siteid reop.csv")
    return dfmort

def groupby_complics_siteid():
    df_comp = df_all.groupby('SiteID')['Complics'].apply(lambda x: (x == 1).sum()).reset_index(name='Complics')
    dfmort = df_reop.groupby('SiteID')['Complics'].apply(lambda x: (x == 1).sum()).reset_index(name='Complics_reop')
    df20 = pd.merge(df_comp, dfmort, on='SiteID', how='outer')
    df20.to_csv("/tmp/pycharm_project_723/files/Complics siteid.csv")
    return df20

def groupby_mortality_surgid():
    dfmort = df_all.groupby('surgid')['MtOpD'].apply(lambda x: (x == 1).sum()).reset_index(name='Mortality')
    dfmort.to_csv("/tmp/pycharm_project_723/files/mortality surgid.csv")
    return dfmort

def groupby_mortality_surgid_reop():
    dfmort = df_reop.groupby('surgid')['MtOpD'].apply(lambda x: (x == 1).sum()).reset_index(name='Mortality_reop')
    dfmort.to_csv("/tmp/pycharm_project_723/files/mortality surgid reop.csv")
    return dfmort

def groupby_complics_surgid():
    df_comp = df_all.groupby('surgid')['Complics'].apply(lambda x: (x == 1).sum()).reset_index(name='Complics')
    dfmort = df_reop.groupby('surgid')['Complics'].apply(lambda x: (x == 1).sum()).reset_index(name='Complics_reop')
    df20 = pd.merge(df_comp, dfmort, on='surgid', how='outer')
    df20.to_csv("/tmp/pycharm_project_723/files/Complics surgid.csv")
    return df20

def launch_reg_siteid():
    df_sites = pd.read_csv("total_avg_site_id.csv")

    print("---------------------site id reg:--------------------------")
    print()
    X = df_sites[['total_year_avg', 'Year_avg_Firstop', 'Year_avg_reop', 'reop/total']]  # here we have 2 variables for multiple regression. If you just want to use one variable for simple linear regression, then use X = df['Interest_Rate'] for example.Alternatively, you may add additional variables within the brackets
    Y = df_sites['mortalty_rate']

    # with sklearn
    regr = linear_model.LinearRegression()
    regr.fit(X, Y)

    print('Intercept: \n', regr.intercept_)
    print('Coefficients: \n', regr.coef_)

       # with statsmodels
    X = sm.add_constant(X)  # adding a constant

    model = sm.OLS(Y, X).fit()
    predictions = model.predict(X)

    print_model = model.summary()
    print(print_model)
    print()
    print()



def launch_reg_surgid():
    df_sites = pd.read_csv("total_avg_surgid.csv")

    print("------------------------ surg id reg:-------------------------------")
    print()
    X = df_sites[['total_year_avg', 'Year_avg_Firstop', 'Year_avg_reop','reop/total']]  # here we have 2 variables for multiple regression. If you just want to use one variable for simple linear regression, then use X = df['Interest_Rate'] for example.Alternatively, you may add additional variables within the brackets
    Y = df_sites['mortalty_rate']

    # with sklearn
    regr = linear_model.LinearRegression()
    regr.fit(X, Y)

    print('Intercept: \n', regr.intercept_)
    print('Coefficients: \n', regr.coef_)

    # with statsmodels
    X = sm.add_constant(X)  # adding a constant

    model = sm.OLS(Y, X).fit()
    predictions = model.predict(X)

    print_model = model.summary()
    print(print_model)

groupby_siteid()
groupby_siteid_reop()

groupby_surgid()
groupby_surgid_reop()

launch_reg_siteid()
launch_reg_surgid()