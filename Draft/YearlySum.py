import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


#-------------read csv---------------------
df_2010_2011 = pd.read_csv("/mnt/nadavrap-students/STS/data/data_Shapira_20200911_2010_2011.csv")
df_2012_2013 = pd.read_csv("/mnt/nadavrap-students/STS/data/data_Shapira_20200911_2012_2013.csv")
df_2014_2015 = pd.read_csv("/mnt/nadavrap-students/STS/data/data_Shapira_20200911_2014_2015.csv")
df_2016_2017 = pd.read_csv("/mnt/nadavrap-students/STS/data/data_Shapira_20200911_2016_2017.csv")
df_2018_2019 = pd.read_csv("/mnt/nadavrap-students/STS/data/data_Shapira_20200911_2018_2019.csv")


df_2010_2011['prcab'].fillna(2, inplace=True)
df_2012_2013['prcab'].fillna(2, inplace=True)
df_2014_2015['prcab'].fillna(2, inplace=True)
df_2016_2017['prcab'].fillna(2, inplace=True)
df_2018_2019['prcab'].fillna(2, inplace=True)


mask = df_2010_2011['surgyear'] != 2010
df_2011 = df_2010_2011[mask]
df_2010 = df_2010_2011[~mask]
mask2 = df_2012_2013['surgyear'] != 2012
df_2013 = df_2012_2013[mask2]
df_2012 = df_2012_2013[~mask2]
mask3 = df_2014_2015['surgyear'] != 2014
df_2015 = df_2014_2015[mask3]
df_2014 = df_2014_2015[~mask3]
mask4 = df_2016_2017['surgyear'] != 2016
df_2017 = df_2016_2017[mask4]
df_2016 = df_2016_2017[~mask4]
mask5 = df_2018_2019['surgyear'] != 2018
df_2019 = df_2018_2019[mask5]
df_2018 = df_2018_2019[~mask5]

avg_siteid = pd.DataFrame()
avg_surgid = pd.DataFrame()

def groupby_siteid():
    df2010 = df_2010.groupby('siteid')['siteid'].count().reset_index(name='2010_total')
    df2011 = df_2011.groupby('siteid')['siteid'].count().reset_index(name='2011_total')
    df2012 = df_2012.groupby('siteid')['siteid'].count().reset_index(name='2012_total')
    df2013 = df_2013.groupby('siteid')['siteid'].count().reset_index(name='2013_total')
    df2014 = df_2014.groupby('siteid')['siteid'].count().reset_index(name='2014_total')
    df2015 = df_2015.groupby('siteid')['siteid'].count().reset_index(name='2015_total')
    df2016 = df_2016.groupby('siteid')['siteid'].count().reset_index(name='2016_total')
    df2017 = df_2017.groupby('siteid')['siteid'].count().reset_index(name='2017_total')
    df2018 = df_2018.groupby('siteid')['siteid'].count().reset_index(name='2018_total')
    df2019 = df_2019.groupby('siteid')['siteid'].count().reset_index(name='2019_total')


    df1 =pd.merge(df2010, df2011, on='siteid', how='outer')
    df2 =pd.merge(df1, df2012, on='siteid', how='outer')
    df3 =pd.merge(df2, df2013, on='siteid', how='outer')
    df4 =pd.merge(df3, df2014, on='siteid', how='outer')
    df5 =pd.merge(df4, df2015, on='siteid', how='outer')
    df6 =pd.merge(df5, df2016, on='siteid', how='outer')
    df7 =pd.merge(df6, df2017, on='siteid', how='outer')
    df8 =pd.merge(df7, df2018, on='siteid', how='outer')
    df_sum_all_Years =pd.merge(df8, df2019, on='siteid', how='outer')

    df_sum_all_Years.fillna(0,inplace=True)

    cols = df_sum_all_Years.columns.difference(['siteid'])
    df_sum_all_Years['Distinct_years'] = df_sum_all_Years[cols].gt(0).sum(axis=1)


    cols_sum = df_sum_all_Years.columns.difference(['siteid','Distinct_years'])
    df_sum_all_Years['Year_sum'] =df_sum_all_Years.loc[:,cols_sum].sum(axis=1)
    df_sum_all_Years['Year_avg'] = df_sum_all_Years['Year_sum']/df_sum_all_Years['Distinct_years']
    df_sum_all_Years.to_csv("total op sum all years siteid.csv")
    print("details on site id dist:")
    print("num of all sites: ", len(df_sum_all_Years))

    less_8 =df_sum_all_Years[df_sum_all_Years['Distinct_years'] !=10]
    less_8.to_csv("total op less 10 years siteid.csv")
    print("num of sites with less years: ", len(less_8))

    x = np.array(less_8['Distinct_years'])
    print(np.unique(x))
    avg_siteid['siteid'] = df_sum_all_Years['siteid']
    avg_siteid['total_year_sum'] = df_sum_all_Years['Year_sum']
    avg_siteid['total_year_avg'] = df_sum_all_Years['Year_avg']
    avg_siteid['num_of_years'] = df_sum_all_Years['Distinct_years']

def groupby_siteid_prcab():
    df2010 = df_2010.groupby('siteid')['prcab'].apply(lambda x: (x == 1).sum()).reset_index(name='2010_reop')
    df2011 = df_2011.groupby('siteid')['prcab'].apply(lambda x: (x == 1).sum()).reset_index(name='2011_reop')
    df2012 = df_2012.groupby('siteid')['prcab'].apply(lambda x: (x == 1).sum()).reset_index(name='2012_reop')
    df2013 = df_2013.groupby('siteid')['prcab'].apply(lambda x: (x == 1).sum()).reset_index(name='2013_reop')
    df2014 = df_2014.groupby('siteid')['prcab'].apply(lambda x: (x == 1).sum()).reset_index(name='2014_reop')
    df2015 = df_2015.groupby('siteid')['prcab'].apply(lambda x: (x == 1).sum()).reset_index(name='2015_reop')
    df2016 = df_2016.groupby('siteid')['prcab'].apply(lambda x: (x == 1).sum()).reset_index(name='2016_reop')
    df2017 = df_2017.groupby('siteid')['prcab'].apply(lambda x: (x == 1).sum()).reset_index(name='2017_reop')
    df2018 = df_2018.groupby('siteid')['prcab'].apply(lambda x: (x == 1).sum()).reset_index(name='2018_reop')
    df2019 = df_2019.groupby('siteid')['prcab'].apply(lambda x: (x == 1).sum()).reset_index(name='2019_reop')

    df1 =pd.merge(df2010, df2011, on='siteid', how='outer')
    df2 =pd.merge(df1, df2012, on='siteid', how='outer')
    df3 =pd.merge(df2, df2013, on='siteid', how='outer')
    df4 =pd.merge(df3, df2014, on='siteid', how='outer')
    df5 =pd.merge(df4, df2015, on='siteid', how='outer')
    df6 =pd.merge(df5, df2016, on='siteid', how='outer')
    df7 =pd.merge(df6, df2017, on='siteid', how='outer')
    df8 =pd.merge(df7, df2018, on='siteid', how='outer')
    df_sum_all_Years =pd.merge(df8, df2019, on='siteid', how='outer')
    df_sum_all_Years.fillna(0,inplace=True)

    cols = df_sum_all_Years.columns.difference(['siteid'])
    df_sum_all_Years['Distinct_years_reop'] = df_sum_all_Years[cols].gt(0).sum(axis=1)

    cols_sum = df_sum_all_Years.columns.difference(['siteid', 'Distinct_years_reop'])
    df_sum_all_Years['Year_sum_reop'] = df_sum_all_Years.loc[:, cols_sum].sum(axis=1)
    df_sum_all_Years['Year_avg_reop'] = df_sum_all_Years['Year_sum_reop'] / avg_siteid['num_of_years']
    df_sum_all_Years.to_csv("sum all years siteid reop.csv")

    less_8 = df_sum_all_Years[df_sum_all_Years['Distinct_years_reop'] != 10]
    less_8.to_csv("less 10 years reop siteid.csv")
    print("num of sites with less years reop : ", len(less_8))

    x = np.array(less_8['Distinct_years_reop'])
    print(np.unique(x))

    df_10 = df_2010.groupby('siteid')['prcab'].apply(lambda x:(x==2).sum()).reset_index(name='2010_Firstop')
    df_11 = df_2011.groupby('siteid')['prcab'].apply(lambda x:(x==2).sum()).reset_index(name='2011_Firstop')
    df_12 = df_2012.groupby('siteid')['prcab'].apply(lambda x:(x==2).sum()).reset_index(name='2012_Firstop')
    df_13 = df_2013.groupby('siteid')['prcab'].apply(lambda x:(x==2).sum()).reset_index(name='2013_Firstop')
    df_14 = df_2014.groupby('siteid')['prcab'].apply(lambda x:(x==2).sum()).reset_index(name='2014_Firstop')
    df_15 = df_2015.groupby('siteid')['prcab'].apply(lambda x:(x==2).sum()).reset_index(name='2015_Firstop')
    df_16 = df_2016.groupby('siteid')['prcab'].apply(lambda x:(x==2).sum()).reset_index(name='2016_Firstop')
    df_17 = df_2017.groupby('siteid')['prcab'].apply(lambda x:(x==2).sum()).reset_index(name='2017_Firstop')
    df_18 = df_2018.groupby('siteid')['prcab'].apply(lambda x:(x==2).sum()).reset_index(name='2018_Firstop')
    df_19 = df_2019.groupby('siteid')['prcab'].apply(lambda x:(x==2).sum()).reset_index(name='2019_Firstop')

    d1 = pd.merge(df_10, df_11, on='siteid', how='outer')
    d2 = pd.merge(d1, df_12, on='siteid', how='outer')
    d3 = pd.merge(d2, df_13, on='siteid', how='outer')
    d4 = pd.merge(d3, df_14, on='siteid', how='outer')
    d5 = pd.merge(d4, df_15, on='siteid', how='outer')
    d6 = pd.merge(d5, df_16, on='siteid', how='outer')
    d7 = pd.merge(d6, df_17, on='siteid', how='outer')
    d8 = pd.merge(d7, df_18, on='siteid', how='outer')
    df_sum_all_Years_total = pd.merge(d8, df_19, on='siteid', how='outer')
    df_sum_all_Years_total.fillna(0, inplace=True)

    cols = df_sum_all_Years_total.columns.difference(['siteid'])
    df_sum_all_Years_total['Distinct_years'] = df_sum_all_Years_total[cols].gt(0).sum(axis=1)

    cols_sum = df_sum_all_Years_total.columns.difference(['siteid', 'Distinct_years'])
    df_sum_all_Years_total['Year_sum'] = df_sum_all_Years_total.loc[:, cols_sum].sum(axis=1)
    df_sum_all_Years_total['Year_avg'] = df_sum_all_Years_total['Year_sum'] / avg_siteid['num_of_years']
    df_sum_all_Years_total.to_csv("First op sum all years siteid.csv")

    less = df_sum_all_Years_total[df_sum_all_Years_total['Distinct_years'] != 10]
    less.to_csv("First op less 10 years siteid.csv")
    print("First op num of sites with less years: ", len(less))

    x = np.array(less['Distinct_years'])
    print(np.unique(x))

    temp_first = pd.DataFrame()
    temp_first['siteid'] = df_sum_all_Years_total['siteid']
    temp_first['Year_sum_Firstop'] = df_sum_all_Years_total['Year_sum']
    temp_first['Year_avg_Firstop'] = df_sum_all_Years_total['Year_avg']
    temp_reop = pd.DataFrame()
    temp_reop['siteid'] = df_sum_all_Years['siteid']
    temp_reop['Year_avg_reop'] = df_sum_all_Years['Year_avg_reop']
    temp_reop['Year_sum_reop'] = df_sum_all_Years['Year_sum_reop']

    df20 = pd.merge(avg_siteid, temp_first, on='siteid', how='outer')
    total_avg_site_id = pd.merge(df20, temp_reop, on='siteid', how='outer')

    total_avg_site_id['firstop/total'] = (total_avg_site_id['Year_avg_Firstop'] / total_avg_site_id['num_of_years']) * 100
    total_avg_site_id['reop/total'] = (total_avg_site_id['Year_avg_reop'] / total_avg_site_id['num_of_years']) * 100
    total_avg_site_id.fillna(0,inplace=True)
    total_avg_site_id.to_csv('total_avg_site_id.csv')

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
    df_sum_all_Years.to_csv("total op sum all years surgid.csv")
    print("details on surg id dist:")
    print("num of all surgid: ", len(df_sum_all_Years))

    less_8 = df_sum_all_Years[df_sum_all_Years['Distinct_years'] != 10]
    less_8.to_csv("total op less 10 years surgid.csv")
    print("num of surgid with less years: ", len(less_8))

    x = np.array(less_8['Distinct_years'])
    print(np.unique(x))
    # avg_surgid['surgid'] = df_sum_all_Years['surgid']
    # avg_surgid['total_year_avg'] = df_sum_all_Years['Year_avg']
    avg_surgid['surgid'] = df_sum_all_Years['surgid']
    avg_surgid['total_year_avg'] = df_sum_all_Years['Year_avg']
    avg_surgid['total_year_count'] = df_sum_all_Years['Year_sum']
    avg_surgid['num_of_years'] = df_sum_all_Years['Distinct_years']

def groupby_surgid_prcab():
    df2010 = df_2010.groupby('surgid')['prcab'].apply(lambda x: (x == 1).sum()).reset_index(name='2010_reop')
    df2011 = df_2011.groupby('surgid')['prcab'].apply(lambda x: (x == 1).sum()).reset_index(name='2011_reop')
    df2012 = df_2012.groupby('surgid')['prcab'].apply(lambda x: (x == 1).sum()).reset_index(name='2012_reop')
    df2013 = df_2013.groupby('surgid')['prcab'].apply(lambda x: (x == 1).sum()).reset_index(name='2013_reop')
    df2014 = df_2014.groupby('surgid')['prcab'].apply(lambda x: (x == 1).sum()).reset_index(name='2014_reop')
    df2015 = df_2015.groupby('surgid')['prcab'].apply(lambda x: (x == 1).sum()).reset_index(name='2015_reop')
    df2016 = df_2016.groupby('surgid')['prcab'].apply(lambda x: (x == 1).sum()).reset_index(name='2016_reop')
    df2017 = df_2017.groupby('surgid')['prcab'].apply(lambda x: (x == 1).sum()).reset_index(name='2017_reop')
    df2018 = df_2018.groupby('surgid')['prcab'].apply(lambda x: (x == 1).sum()).reset_index(name='2018_reop')
    df2019 = df_2019.groupby('surgid')['prcab'].apply(lambda x: (x == 1).sum()).reset_index(name='2019_reop')

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
    df_sum_all_Years.to_csv("sum all years surgid reop.csv")

    less_8 = df_sum_all_Years[df_sum_all_Years['Distinct_years_reop'] != 10]
    less_8.to_csv("less 10 years reop surgid.csv")
    print("num of surgid with less years reop : ", len(less_8))

    x = np.array(less_8['Distinct_years_reop'])
    print(np.unique(x))

    df_10 = df_2010.groupby('surgid')['prcab'].apply(lambda x: (x == 2).sum()).reset_index(name='2010_Firstop')
    df_11 = df_2011.groupby('surgid')['prcab'].apply(lambda x: (x == 2).sum()).reset_index(name='2011_Firstop')
    df_12 = df_2012.groupby('surgid')['prcab'].apply(lambda x: (x == 2).sum()).reset_index(name='2012_Firstop')
    df_13 = df_2013.groupby('surgid')['prcab'].apply(lambda x: (x == 2).sum()).reset_index(name='2013_Firstop')
    df_14 = df_2014.groupby('surgid')['prcab'].apply(lambda x: (x == 2).sum()).reset_index(name='2014_Firstop')
    df_15 = df_2015.groupby('surgid')['prcab'].apply(lambda x: (x == 2).sum()).reset_index(name='2015_Firstop')
    df_16 = df_2016.groupby('surgid')['prcab'].apply(lambda x: (x == 2).sum()).reset_index(name='2016_Firstop')
    df_17 = df_2017.groupby('surgid')['prcab'].apply(lambda x: (x == 2).sum()).reset_index(name='2017_Firstop')
    df_18 = df_2018.groupby('surgid')['prcab'].apply(lambda x: (x == 2).sum()).reset_index(name='2018_Firstop')
    df_19 = df_2019.groupby('surgid')['prcab'].apply(lambda x: (x == 2).sum()).reset_index(name='2019_Firstop')

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
    df_sum_all_Years_total.to_csv("First op sum all years surgid.csv")

    less = df_sum_all_Years_total[df_sum_all_Years_total['Distinct_years'] != 10]
    less.to_csv("First op less 10 years surgid.csv")
    print("First op num of sites with less years: ", len(less))

    x = np.array(less['Distinct_years'])
    print(np.unique(x))

    # temp_first = pd.DataFrame()
    # temp_first['surgid'] = df_sum_all_Years_total['surgid']
    # temp_first['Year_avg_Firstop'] = df_sum_all_Years_total['Year_avg']
    # temp_reop = pd.DataFrame()
    # temp_reop['surgid'] = df_sum_all_Years['surgid']
    # temp_reop['Year_avg_reop'] = df_sum_all_Years['Year_avg_reop']
    #
    # df20 = pd.merge(avg_surgid, temp_first, on='surgid', how='outer')
    # total_avg_surgid = pd.merge(df20, temp_reop, on='surgid', how='outer')
    #
    # total_avg_surgid['firstop/total'] = (total_avg_surgid['Year_avg_Firstop'] / total_avg_surgid['total_year_avg']) * 100
    # total_avg_surgid['reop/total'] = (total_avg_surgid['Year_avg_reop'] / total_avg_surgid['total_year_avg']) * 100
    # total_avg_surgid.fillna(0, inplace=True)
    # total_avg_surgid.to_csv('total_avg_surgid.csv')


    temp_first = pd.DataFrame()
    temp_first['surgid'] = df_sum_all_Years_total['surgid']
    temp_first['Year_avg_Firstop'] = df_sum_all_Years_total['Year_avg']
    temp_first['Year_sum_Firstop'] = df_sum_all_Years_total['Year_sum']
    temp_reop = pd.DataFrame()
    temp_reop['surgid'] = df_sum_all_Years['surgid']
    temp_reop['Year_avg_reop'] = df_sum_all_Years['Year_avg_reop']
    temp_reop['Year_sum_reop'] = df_sum_all_Years['Year_sum_reop']

    df20 = pd.merge(avg_surgid, temp_first, on='surgid', how='outer')
    total_avg_surgid = pd.merge(df20, temp_reop, on='surgid', how='outer')

    total_avg_surgid['firstop/total'] = (total_avg_surgid['Year_sum_Firstop'] / total_avg_surgid['total_year_count']) * 100
    total_avg_surgid['reop/total'] = (total_avg_surgid['Year_sum_reop'] / total_avg_surgid['total_year_count']) * 100
    total_avg_surgid.fillna(0, inplace=True)
    total_avg_surgid.to_csv('total_avg_surgid.csv')

def draw_hist(data,num_of_bins,title,x_title,y_title,color):
    plt.hist(data, bins=num_of_bins, color=color,ec="black")
    plt.title(title)
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.show()


groupby_siteid()
groupby_siteid_prcab()
groupby_surgid()
groupby_surgid_prcab()


df_avg_siteid = pd.read_csv("total_avg_site_id.csv")
df_avg_surgid = pd.read_csv("total_avg_surgid.csv")
# df_avg2_surgid = pd.read_csv("total_avg_surgid_sum avg count.csv")
# # # df_sum_hospid= pd.read_csv(path+"sum all years hospid.csv")
# #
# #
# draw_hist(df_avg_siteid['total_year_avg'],40,"siteid Histogram of yearly avg operation",'avg of Operation',"count of siteid",'skyblue')
# draw_hist(df_avg_siteid['Year_avg_Firstop'],40,"siteid Histogram of yearly avg First operation",'avg of First Operation',"count of siteid",'skyblue')
# draw_hist(df_avg_siteid['Year_avg_reop'],40,"siteid Histogram of yearly avg reOperation",'avg of reOperation',"count of siteid",'skyblue')
#
# draw_hist(df_avg_siteid['firstop/total'],40,"siteid Histogram of yearly avg First operation/Total operation",'% of First Operation',"count of siteid",'palegreen')
# draw_hist(df_avg_siteid['reop/total'],40,"siteid Histogram of yearly avg reOperation/Total operation",'% of reOperation',"count of siteid",'palegreen')
#
# # draw_hist(df_sum_surgid['Year_avg'],20,"surgid Histogram of yearly avg operation",'avg of Operation',"count of surgid")
# draw_hist(df_avg_surgid['total_year_avg'],40,"surgid Histogram of yearly avg operation",'avg of Operation',"count of surgid",'plum')
# draw_hist(df_avg_surgid['Year_avg_Firstop'],40,"surgid Histogram of yearly avg First operation",'avg of First Operation',"count of surgid",'plum')
# draw_hist(df_avg_surgid['Year_avg_reop'],40,"surgid Histogram of yearly avg reOperation",'avg of reOperation',"count of surgid",'plum')
#
# draw_hist(df_avg_surgid['firstop/total'],40,"surgid Histogram of yearly avg First operation/Total operation",'% of First Operation',"count of surgid",'bisque')
# draw_hist(df_avg_surgid['reop/total'],40,"surgid Histogram of yearly avg reOperation/Total operation",'% of reOperation',"count of surgid",'bisque')

#
# draw_hist(df_avg2_surgid['total_year_avg'],40,"surgid Histogram of yearly avg operation",'avg of Operation',"count of surgid",'plum')
# draw_hist(df_avg2_surgid['Year_avg_Firstop'],40,"surgid Histogram of yearly avg First operation",'avg of First Operation',"count of surgid",'plum')
# draw_hist(df_avg2_surgid['Year_avg_reop'],40,"surgid Histogram of yearly avg reOperation",'avg of reOperation',"count of surgid",'plum')
#
# draw_hist(df_avg2_surgid['firstop/total'],40,"surgid Histogram of yearly avg First operation/Total operation",'% of First Operation',"count of surgid",'bisque')
# draw_hist(df_avg2_surgid['reop/total'],40,"surgid Histogram of yearly avg reOperation/Total operation",'% of reOperation',"count of surgid",'bisque')