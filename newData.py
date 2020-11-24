import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#-------------read csv---------------------
df_2010_2011 = pd.read_csv("/mnt/nadavrap-students/STS/data/data_Shapira_20200911_2010_2011.csv")
df_2012_2013 = pd.read_csv("/mnt/nadavrap-students/STS/data/data_Shapira_20200911_2012_2013.csv")
df_2014_2015 = pd.read_csv("/mnt/nadavrap-students/STS/data/data_Shapira_20200911_2014_2015.csv")
df_2016_2017 = pd.read_csv("/mnt/nadavrap-students/STS/data/data_Shapira_20200911_2016_2017.csv")
df_2018_2019 = pd.read_csv("/mnt/nadavrap-students/STS/data/data_Shapira_20200911_2018_2019.csv")

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
# #tmpHilla=df_2018_2019.columns
# tmpHilla=pd.DataFrame(df_2018_2019.columns.values.tolist())
# tmpHilla.to_csv("/tmp/pycharm_project_355/columns.csv")

# my_list = df_2010_2011.columns.values.tolist()
# print (my_list)
# print()
# my_list = df_2012_2013.columns.values.tolist()
# print (my_list)
# print()
# my_list = df_2014_2015.columns.values.tolist()
# print (my_list)
# print()
# my_list = df_2016_2017.columns.values.tolist()
# print (my_list)
# print()
# my_list = df_2018_2019.columns.values.tolist()
# print (my_list)
# print()

#-------------------merge all csv--------------------------
# dfMerge1 = pd.merge(df_2010_2011, df_2012_2013, on='surgorder')
# dfMerge2 = pd.merge(dfMerge1, df_2014_2015, on='surgorder')
# dfMerge = pd.merge(dfMerge2, df_2016_2017, on='surgorder')
#dfMerge = pd.merge(df_2010_2011, df_2012_2013, on='SiteID')
#count distinc
#table.groupby('YEARMONTH').CLIENTCODE.nunique()

def groupby_siteid():
    df_2010 = df_2010_2011.groupby('siteid')['surgyear'].apply(lambda x: (x== 2010 ).sum()).reset_index(name='2010')
    df_2011 = df_2010_2011.groupby('siteid')['surgyear'].apply(lambda x: (x== 2011 ).sum()).reset_index(name='2011')
    df_2012 = df_2012_2013.groupby('siteid')['surgyear'].apply(lambda x: (x== 2012 ).sum()).reset_index(name='2012')
    df_2013 = df_2012_2013.groupby('siteid')['surgyear'].apply(lambda x: (x== 2013 ).sum()).reset_index(name='2013')
    df_2014 = df_2014_2015.groupby('siteid')['surgyear'].apply(lambda x: (x== 2014 ).sum()).reset_index(name='2014')
    df_2015 = df_2014_2015.groupby('siteid')['surgyear'].apply(lambda x: (x== 2015 ).sum()).reset_index(name='2015')
    df_2016 = df_2016_2017.groupby('siteid')['surgyear'].apply(lambda x: (x== 2016 ).sum()).reset_index(name='2016')
    df_2017 = df_2016_2017.groupby('siteid')['surgyear'].apply(lambda x: (x== 2017 ).sum()).reset_index(name='2017')
    df_2018 = df_2018_2019.groupby('siteid')['surgyear'].apply(lambda x: (x== 2018 ).sum()).reset_index(name='2018')
    df_2019 = df_2018_2019.groupby('siteid')['surgyear'].apply(lambda x: (x== 2019 ).sum()).reset_index(name='2019')


    df1 =pd.merge(df_2010, df_2011, on='siteid')
    df2 =pd.merge(df1, df_2012, on='siteid')
    df3 =pd.merge(df2, df_2013, on='siteid')
    df4 =pd.merge(df3, df_2014, on='siteid')
    df5 =pd.merge(df4, df_2015, on='siteid')
    df6 =pd.merge(df5, df_2016, on='siteid')
    df7 =pd.merge(df6, df_2017, on='siteid')
    df8 =pd.merge(df7, df_2018, on='siteid')
    df_sum_all_Years =pd.merge(df8, df_2019, on='siteid')

    cols = df_sum_all_Years.columns.difference(['siteid'])
    df_sum_all_Years['Distinct_years'] = df_sum_all_Years[cols].gt(0).sum(axis=1)


    cols_sum = df_sum_all_Years.columns.difference(['siteid','Distinct_years'])
    df_sum_all_Years['Year_sum'] =df_sum_all_Years.loc[:,cols_sum].sum(axis=1)
    df_sum_all_Years['Year_avg'] = df_sum_all_Years['Year_sum']/df_sum_all_Years['Distinct_years']
    df_sum_all_Years.to_csv("total op sum all years siteid.csv")
    print("details on site id dist:")
    print ("num of all sites: ", len(df_sum_all_Years))

    less_8 =df_sum_all_Years[df_sum_all_Years['Distinct_years'] !=10]
    less_8.to_csv("total op less 10 years siteid.csv")
    print("num of sites with less years: ", len(less_8))

    x = np.array(less_8['Distinct_years'])
    print(np.unique(x))
    avg_siteid['siteid'] = df_sum_all_Years['siteid']
    avg_siteid['total_year_avg'] = df_sum_all_Years['Year_avg']


def groupby_surgid():
    df_2010 = df_2010_2011.groupby('surgid')['surgyear'].apply(lambda x: (x== 2010 ).sum()).reset_index(name='2010')
    df_2011 = df_2010_2011.groupby('surgid')['surgyear'].apply(lambda x: (x== 2011 ).sum()).reset_index(name='2011')
    df_2012 = df_2012_2013.groupby('surgid')['surgyear'].apply(lambda x: (x== 2012 ).sum()).reset_index(name='2012')
    df_2013 = df_2012_2013.groupby('surgid')['surgyear'].apply(lambda x: (x== 2013 ).sum()).reset_index(name='2013')
    df_2014 = df_2014_2015.groupby('surgid')['surgyear'].apply(lambda x: (x== 2014 ).sum()).reset_index(name='2014')
    df_2015 = df_2014_2015.groupby('surgid')['surgyear'].apply(lambda x: (x== 2015 ).sum()).reset_index(name='2015')
    df_2016 = df_2016_2017.groupby('surgid')['surgyear'].apply(lambda x: (x== 2016 ).sum()).reset_index(name='2016')
    df_2017 = df_2016_2017.groupby('surgid')['surgyear'].apply(lambda x: (x== 2017 ).sum()).reset_index(name='2017')
    df_2018 = df_2018_2019.groupby('surgid')['surgyear'].apply(lambda x: (x== 2018 ).sum()).reset_index(name='2018')
    df_2019 = df_2018_2019.groupby('surgid')['surgyear'].apply(lambda x: (x== 2019 ).sum()).reset_index(name='2019')


    df1 =pd.merge(df_2010, df_2011, on='surgid')
    df2 =pd.merge(df1, df_2012, on='surgid')
    df3 =pd.merge(df2, df_2013, on='surgid')
    df4 =pd.merge(df3, df_2014, on='surgid')
    df5 =pd.merge(df4, df_2015, on='surgid')
    df6 =pd.merge(df5, df_2016, on='surgid')
    df7 =pd.merge(df6, df_2017, on='surgid')
    df8 =pd.merge(df7, df_2018, on='surgid')
    df_sum_all_Years =pd.merge(df8, df_2019, on='surgid')

    cols = df_sum_all_Years.columns.difference(['surgid'])
    df_sum_all_Years['Distinct_years'] = df_sum_all_Years[cols].gt(0).sum(axis=1)


    cols_sum = df_sum_all_Years.columns.difference(['surgid','Distinct_years'])
    df_sum_all_Years['Year_sum'] =df_sum_all_Years.loc[:,cols_sum].sum(axis=1)
    df_sum_all_Years['Year_avg'] = df_sum_all_Years['Year_sum']/df_sum_all_Years['Distinct_years']
    df_sum_all_Years.to_csv("sum all years surgid.csv")
    print()
    print("details of surgid dist:")
    print("num of all surgid: ", len(df_sum_all_Years))

    less_8 =df_sum_all_Years[df_sum_all_Years['Distinct_years'] !=10]
    less_8.to_csv("less 10 years surgid.csv")
    print("num of doctors with less years: ", len(less_8))

    x = np.array(less_8['Distinct_years'])
    print(np.unique(x))
    avg_surgid['surgid'] = df_sum_all_Years['surgid']
    avg_surgid['total_year_avg'] = df_sum_all_Years['Year_avg']

def groupby_hospid():
    df_2010 = df_2010_2011.groupby('hospid')['surgyear'].apply(lambda x: (x== 2010 ).sum()).reset_index(name='2010')
    df_2011 = df_2010_2011.groupby('hospid')['surgyear'].apply(lambda x: (x== 2011 ).sum()).reset_index(name='2011')
    df_2012 = df_2012_2013.groupby('hospid')['surgyear'].apply(lambda x: (x== 2012 ).sum()).reset_index(name='2012')
    df_2013 = df_2012_2013.groupby('hospid')['surgyear'].apply(lambda x: (x== 2013 ).sum()).reset_index(name='2013')
    df_2014 = df_2014_2015.groupby('hospid')['surgyear'].apply(lambda x: (x== 2014 ).sum()).reset_index(name='2014')
    df_2015 = df_2014_2015.groupby('hospid')['surgyear'].apply(lambda x: (x== 2015 ).sum()).reset_index(name='2015')
    df_2016 = df_2016_2017.groupby('hospid')['surgyear'].apply(lambda x: (x== 2016 ).sum()).reset_index(name='2016')
    df_2017 = df_2016_2017.groupby('hospid')['surgyear'].apply(lambda x: (x== 2017 ).sum()).reset_index(name='2017')
    df_2018 = df_2018_2019.groupby('hospid')['surgyear'].apply(lambda x: (x== 2018 ).sum()).reset_index(name='2018')
    df_2019 = df_2018_2019.groupby('hospid')['surgyear'].apply(lambda x: (x== 2019 ).sum()).reset_index(name='2019')


    df1 =pd.merge(df_2010, df_2011, on='hospid')
    df2 =pd.merge(df1, df_2012, on='hospid')
    df3 =pd.merge(df2, df_2013, on='hospid')
    df4 =pd.merge(df3, df_2014, on='hospid')
    df5 =pd.merge(df4, df_2015, on='hospid')
    df6 =pd.merge(df5, df_2016, on='hospid')
    df7 =pd.merge(df6, df_2017, on='hospid')
    df8 =pd.merge(df7, df_2018, on='hospid')
    df_sum_all_Years =pd.merge(df8, df_2019, on='hospid')

    cols = df_sum_all_Years.columns.difference(['hospid'])
    df_sum_all_Years['Distinct_years'] = df_sum_all_Years[cols].gt(0).sum(axis=1)


    cols_sum = df_sum_all_Years.columns.difference(['hospid','Distinct_years'])
    df_sum_all_Years['Year_sum'] =df_sum_all_Years.loc[:,cols_sum].sum(axis=1)
    df_sum_all_Years['Year_avg'] = df_sum_all_Years['Year_sum']/df_sum_all_Years['Distinct_years']
    df_sum_all_Years.to_csv("sum all years hospid.csv")
    print(df_sum_all_Years)
    print ("num of all sites: ", len(df_sum_all_Years))

    less_8 =df_sum_all_Years[df_sum_all_Years['Distinct_years'] !=10]
    less_8.to_csv("less 10 years hospid.csv")
    print("num of hospital with less years: ", len(less_8))

    x = np.array(less_8['Distinct_years'])
    print(np.unique(x))
    return df_sum_all_Years

def draw_hist(data,num_of_bins,title,x_title,y_title,color):
    plt.hist(data, bins=num_of_bins, color=color,ec="black")
    plt.title(title)
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.show()

def group_by_count(group_by_value,name):
    df_2010_2011_gb = df_2010_2011.groupby(group_by_value)[group_by_value].count().reset_index(name=name)
    df_2012_2013_gb = df_2012_2013.groupby(group_by_value)[group_by_value].count().reset_index(name=name)
    df_2014_2015_gb = df_2014_2015.groupby(group_by_value)[group_by_value].count().reset_index(name=name)
    df_2016_2017_gb = df_2016_2017.groupby(group_by_value)[group_by_value].count().reset_index(name=name)
    df_2018_2019_gb = df_2018_2019.groupby(group_by_value)[group_by_value].count().reset_index(name=name)

    df_merge_1=pd.merge(df_2010_2011_gb,df_2012_2013_gb, on=group_by_value)
    df_merge_2=pd.merge(df_merge_1,df_2014_2015_gb, on=group_by_value)
    df_merge_3=pd.merge(df_merge_2,df_2016_2017_gb, on=group_by_value)
    df_merge_4=pd.merge(df_merge_3,df_2018_2019_gb, on=group_by_value)

    cols = df_merge_4.columns.difference([group_by_value])
    df_merge_4[name] = df_merge_4.loc[:,cols].sum(axis=1)

    df_new=pd.DataFrame()
    df_new[group_by_value] = df_merge_4[group_by_value]
    df_new[name] = df_merge_4[name]

    return df_new


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


    df1 = pd.merge(df2010, df2011, on='siteid')
    df2 = pd.merge(df1, df2012, on='siteid')
    df3 = pd.merge(df2, df2013, on='siteid')
    df4 = pd.merge(df3, df2014, on='siteid')
    df5 = pd.merge(df4, df2015, on='siteid')
    df6 = pd.merge(df5, df2016, on='siteid')
    df7 = pd.merge(df6, df2017, on='siteid')
    df8 = pd.merge(df7, df2018, on='siteid')
    df_sum_all_Years = pd.merge(df8, df2019, on='siteid')

    cols = df_sum_all_Years.columns.difference(['siteid'])
    df_sum_all_Years['Distinct_years_reop'] = df_sum_all_Years[cols].gt(0).sum(axis=1)

    cols_sum = df_sum_all_Years.columns.difference(['siteid', 'Distinct_years_reop'])
    df_sum_all_Years['Year_sum_reop'] = df_sum_all_Years.loc[:, cols_sum].sum(axis=1)
    df_sum_all_Years['Year_avg_reop'] = df_sum_all_Years['Year_sum_reop'] / df_sum_all_Years['Distinct_years_reop']
    df_sum_all_Years.to_csv("sum all years siteid reop.csv")

    less_8 = df_sum_all_Years[df_sum_all_Years['Distinct_years_reop'] != 10]
    less_8.to_csv("less 10 years reop siteid.csv")
    print("num of sites with less years: ", len(less_8))

    x = np.array(less_8['Distinct_years_reop'])
    print(np.unique(x))

    df_10 = df_2010.groupby('siteid')['prcab'].apply(lambda x: (x == 2).sum()).reset_index(name='2010_Firstop')
    df_11 = df_2011.groupby('siteid')['prcab'].apply(lambda x: (x == 2).sum()).reset_index(name='2011_Firstop')
    df_12 = df_2012.groupby('siteid')['prcab'].apply(lambda x: (x == 2).sum()).reset_index(name='2012_Firstop')
    df_13 = df_2013.groupby('siteid')['prcab'].apply(lambda x: (x == 2).sum()).reset_index(name='2013_Firstop')
    df_14 = df_2014.groupby('siteid')['prcab'].apply(lambda x: (x == 2).sum()).reset_index(name='2014_Firstop')
    df_15 = df_2015.groupby('siteid')['prcab'].apply(lambda x: (x == 2).sum()).reset_index(name='2015_Firstop')
    df_16 = df_2016.groupby('siteid')['prcab'].apply(lambda x: (x == 2).sum()).reset_index(name='2016_Firstop')
    df_17 = df_2017.groupby('siteid')['prcab'].apply(lambda x: (x == 2).sum()).reset_index(name='2017_Firstop')
    df_18 = df_2018.groupby('siteid')['prcab'].apply(lambda x: (x == 2).sum()).reset_index(name='2018_Firstop')
    df_19 = df_2019.groupby('siteid')['prcab'].apply(lambda x: (x == 2).sum()).reset_index(name='2019_Firstop')

    d1 = pd.merge(df_10, df_11, on='siteid')
    d2 = pd.merge(d1, df_12, on='siteid')
    d3 = pd.merge(d2, df_13, on='siteid')
    d4 = pd.merge(d3, df_14, on='siteid')
    d5 = pd.merge(d4, df_15, on='siteid')
    d6 = pd.merge(d5, df_16, on='siteid')
    d7 = pd.merge(d6, df_17, on='siteid')
    d8 = pd.merge(d7, df_18, on='siteid')
    df_sum_all_Years_total = pd.merge(d8, df_19, on='siteid')
    cols = df_sum_all_Years_total.columns.difference(['siteid'])
    df_sum_all_Years_total['Distinct_years'] = df_sum_all_Years_total[cols].gt(0).sum(axis=1)

    cols_sum = df_sum_all_Years_total.columns.difference(['siteid', 'Distinct_years'])
    df_sum_all_Years_total['Year_sum'] = df_sum_all_Years_total.loc[:, cols_sum].sum(axis=1)
    df_sum_all_Years_total['Year_avg'] = df_sum_all_Years_total['Year_sum'] / df_sum_all_Years_total['Distinct_years']
    df_sum_all_Years_total.to_csv("First op sum all years siteid.csv")

    # df_sum_all_Years.to_csv("sum all years siteid.csv")
    # print(df_sum_all_Years)
    # print("num of all sites: ", len(df_sum_all_Years))
    #
    less = df_sum_all_Years_total[df_sum_all_Years_total['Distinct_years'] != 10]
    less.to_csv("First op less 10 years siteid.csv")
    print("First op num of sites with less years: ", len(less))

    x = np.array(less['Distinct_years'])
    print(np.unique(x))


    temp_first = pd.DataFrame()
    temp_first['siteid'] = df_sum_all_Years_total['siteid']
    temp_first['Year_avg_Firstop'] = df_sum_all_Years_total['Year_avg']
    temp_reop = pd.DataFrame()
    temp_reop['siteid'] = df_sum_all_Years['siteid']
    temp_reop['Year_avg_reop'] = df_sum_all_Years['Year_avg_reop']

    df20 = pd.merge(avg_siteid, temp_first, on='siteid', how='left')
    total_avg_site_id = pd.merge(df20, temp_reop,on='siteid', how='left' )

    total_avg_site_id['firstop/total'] = (total_avg_site_id['Year_avg_Firstop']/total_avg_site_id['total_year_avg'])*100
    total_avg_site_id['reop/total'] = (total_avg_site_id['Year_avg_reop']/total_avg_site_id['total_year_avg'])*100
    total_avg_site_id.to_csv('total_avg_site_id.csv')
    # avg_siteid['Year_avg_Firstop'] = df_sum_all_Years_total['Year_avg']
    # avg_siteid['Year_avg_reop'] = df_sum_all_Years['Year_avg_reop']

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


    df1 = pd.merge(df2010, df2011, on='surgid')
    df2 = pd.merge(df1, df2012, on='surgid')
    df3 = pd.merge(df2, df2013, on='surgid')
    df4 = pd.merge(df3, df2014, on='surgid')
    df5 = pd.merge(df4, df2015, on='surgid')
    df6 = pd.merge(df5, df2016, on='surgid')
    df7 = pd.merge(df6, df2017, on='surgid')
    df8 = pd.merge(df7, df2018, on='surgid')
    df_sum_all_Years = pd.merge(df8, df2019, on='surgid')

    cols = df_sum_all_Years.columns.difference(['surgid'])
    df_sum_all_Years['Distinct_years_reop'] = df_sum_all_Years[cols].gt(0).sum(axis=1)

    cols_sum = df_sum_all_Years.columns.difference(['surgid', 'Distinct_years_reop'])
    df_sum_all_Years['Year_sum_reop'] = df_sum_all_Years.loc[:, cols_sum].sum(axis=1)
    df_sum_all_Years['Year_avg_reop'] = df_sum_all_Years['Year_sum_reop'] / df_sum_all_Years['Distinct_years_reop']
    df_sum_all_Years.to_csv("sum all years surgid reop.csv")

    less_8 = df_sum_all_Years[df_sum_all_Years['Distinct_years_reop'] != 10]
    less_8.to_csv("less 10 years reop surgid.csv")
    print("num of surgid with less years: ", len(less_8))

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

    d1 = pd.merge(df_10, df_11, on='surgid')
    d2 = pd.merge(d1, df_12, on='surgid')
    d3 = pd.merge(d2, df_13, on='surgid')
    d4 = pd.merge(d3, df_14, on='surgid')
    d5 = pd.merge(d4, df_15, on='surgid')
    d6 = pd.merge(d5, df_16, on='surgid')
    d7 = pd.merge(d6, df_17, on='surgid')
    d8 = pd.merge(d7, df_18, on='surgid')
    df_sum_all_Years_total = pd.merge(d8, df_19, on='surgid')
    cols = df_sum_all_Years_total.columns.difference(['surgid'])
    df_sum_all_Years_total['Distinct_years'] = df_sum_all_Years_total[cols].gt(0).sum(axis=1)

    cols_sum = df_sum_all_Years_total.columns.difference(['surgid', 'Distinct_years'])
    df_sum_all_Years_total['Year_sum'] = df_sum_all_Years_total.loc[:, cols_sum].sum(axis=1)
    df_sum_all_Years_total['Year_avg'] = df_sum_all_Years_total['Year_sum'] / df_sum_all_Years_total['Distinct_years']
    df_sum_all_Years_total.to_csv("First op sum all years surgid.csv")

    # df_sum_all_Years.to_csv("sum all years siteid.csv")
    # print(df_sum_all_Years)
    # print("num of all sites: ", len(df_sum_all_Years))
    #
    less = df_sum_all_Years_total[df_sum_all_Years_total['Distinct_years'] != 10]
    less.to_csv("First op less 10 years surgid.csv")
    print("First op num of surgid with less years: ", len(less))

    x = np.array(less['Distinct_years'])
    print(np.unique(x))

    temp_first = pd.DataFrame()
    temp_first['surgid'] = df_sum_all_Years_total['surgid']
    temp_first['Year_avg_Firstop'] = df_sum_all_Years_total['Year_avg']
    temp_reop = pd.DataFrame()
    temp_reop['surgid'] = df_sum_all_Years['surgid']
    temp_reop['Year_avg_reop'] = df_sum_all_Years['Year_avg_reop']

    df20 = pd.merge(avg_surgid, temp_first, on='surgid', how='left')
    total_avg_surgid = pd.merge(df20, temp_reop, on='surgid', how='left')


    total_avg_surgid['firstop/total'] = (total_avg_surgid['Year_avg_Firstop']/total_avg_surgid['total_year_avg'])*100
    total_avg_surgid['reop/total'] = (total_avg_surgid['Year_avg_reop']/total_avg_surgid['total_year_avg'])*100
    total_avg_surgid.to_csv('total_avg_surgid.csv')


groupby_siteid()
# groupby_hospid()
groupby_siteid_prcab()
groupby_surgid()
groupby_surgid_prcab()
#
path="/tmp/pycharm_project_723/"
#
#
# avg_surgid['firstop/total'] = (avg_surgid['Year_avg_Firstop']/avg_surgid['total_year_avg'])*100
# avg_surgid['reop/total'] = (avg_surgid['Year_avg_reop']/avg_surgid['total_year_avg'])*100
#
#
# avg_siteid['firstop/total'] = (avg_siteid['Year_avg_Firstop']/avg_siteid['total_year_avg'])*100
# avg_siteid['reop/total'] = (avg_siteid['Year_avg_reop']/avg_siteid['total_year_avg'])*100
#
# avg_siteid.to_csv('total_avg_site_id.csv')
# avg_surgid.to_csv('total_avg_surgid.csv')


df_avg_siteid = pd.read_csv("total_avg_site_id.csv")
df_avg_surgid = pd.read_csv("total_avg_surgid.csv")
# # df_sum_hospid= pd.read_csv(path+"sum all years hospid.csv")
#
#
draw_hist(df_avg_siteid['total_year_avg'],40,"siteid Histogram of yearly avg operation",'avg of Operation',"count of siteid",'skyblue')
draw_hist(df_avg_siteid['Year_avg_Firstop'].dropna(),40,"siteid Histogram of yearly avg First operation",'avg of First Operation',"count of siteid",'skyblue')
draw_hist(df_avg_siteid['Year_avg_reop'].dropna(),40,"siteid Histogram of yearly avg reOperation",'avg of reOperation',"count of siteid",'skyblue')

draw_hist(df_avg_siteid['firstop/total'].dropna(),40,"siteid Histogram of yearly avg First operation/Total operation",'% of First Operation',"count of siteid",'palegreen')
draw_hist(df_avg_siteid['reop/total'].dropna(),40,"siteid Histogram of yearly avg reOperation/Total operation",'% of reOperation',"count of siteid",'palegreen')

# draw_hist(df_sum_surgid['Year_avg'],20,"surgid Histogram of yearly avg operation",'avg of Operation',"count of surgid")
draw_hist(df_avg_surgid['total_year_avg'],40,"surgid Histogram of yearly avg operation",'avg of Operation',"count of surgid",'plum')
draw_hist(df_avg_surgid['Year_avg_Firstop'].dropna(),40,"surgid Histogram of yearly avg First operation",'avg of First Operation',"count of surgid",'plum')
draw_hist(df_avg_surgid['Year_avg_reop'].dropna(),40,"surgid Histogram of yearly avg reOperation",'avg of reOperation',"count of surgid",'plum')

draw_hist(df_avg_surgid['firstop/total'].dropna(),40,"surgid Histogram of yearly avg First operation/Total operation",'% of First Operation',"count of surgid",'bisque')
draw_hist(df_avg_surgid['reop/total'].dropna(),40,"surgid Histogram of yearly avg reOperation/Total operation",'% of reOperation',"count of surgid",'bisque')