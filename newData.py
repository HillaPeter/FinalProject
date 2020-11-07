import pandas as pd
import matplotlib.pyplot as plt

#-------------read csv---------------------
df_2010_2011 = pd.read_csv("/mnt/nadavrap-students/STS/data/data_Shapira_20200911_2010_2011.csv")
df_2012_2013 = pd.read_csv("/mnt/nadavrap-students/STS/data/data_Shapira_20200911_2012_2013.csv")
df_2014_2015 = pd.read_csv("/mnt/nadavrap-students/STS/data/data_Shapira_20200911_2014_2015.csv")
df_2016_2017 = pd.read_csv("/mnt/nadavrap-students/STS/data/data_Shapira_20200911_2016_2017.csv")
# df_2018_2019 = pd.read_csv("/mnt/nadavrap-students/STS/data/data_Shapira_20200911_2018_2019.csv")

my_list = df_2010_2011.columns.values.tolist()
print (my_list)
print()
my_list = df_2012_2013.columns.values.tolist()
print (my_list)
print()
my_list = df_2014_2015.columns.values.tolist()
print (my_list)
print()
my_list = df_2016_2017.columns.values.tolist()
print (my_list)
print()

#-------------------merge all csv--------------------------
# dfMerge1 = pd.merge(df_2010_2011, df_2012_2013, on='surgorder')
# dfMerge2 = pd.merge(dfMerge1, df_2014_2015, on='surgorder')
# dfMerge = pd.merge(dfMerge2, df_2016_2017, on='surgorder')
#dfMerge = pd.merge(df_2010_2011, df_2012_2013, on='SiteID')
#count distinc
#table.groupby('YEARMONTH').CLIENTCODE.nunique()


df_2010 = df_2010_2011.groupby('siteid')['surgyear'].apply(lambda x: (x== 2010 ).sum()).reset_index(name='2010')
df_2011 = df_2010_2011.groupby('siteid')['surgyear'].apply(lambda x: (x== 2011 ).sum()).reset_index(name='2011')
df_2012 = df_2012_2013.groupby('siteid')['surgyear'].apply(lambda x: (x== 2012 ).sum()).reset_index(name='2012')
df_2013 = df_2012_2013.groupby('siteid')['surgyear'].apply(lambda x: (x== 2013 ).sum()).reset_index(name='2013')
df_2014 = df_2014_2015.groupby('siteid')['surgyear'].apply(lambda x: (x== 2014 ).sum()).reset_index(name='2014')
df_2015 = df_2014_2015.groupby('siteid')['surgyear'].apply(lambda x: (x== 2015 ).sum()).reset_index(name='2015')
df_2016 = df_2016_2017.groupby('siteid')['surgyear'].apply(lambda x: (x== 2016 ).sum()).reset_index(name='2016')
df_2017 = df_2016_2017.groupby('siteid')['surgyear'].apply(lambda x: (x== 2017 ).sum()).reset_index(name='2017')


df1 =pd.merge(df_2010, df_2011, on='siteid')
df2 =pd.merge(df1, df_2012, on='siteid')
df3 =pd.merge(df2, df_2013, on='siteid')
df4 =pd.merge(df3, df_2014, on='siteid')
df5 =pd.merge(df4, df_2015, on='siteid')
df6 =pd.merge(df5, df_2016, on='siteid')
df_sum_all_Years =pd.merge(df6, df_2017, on='siteid')

cols = df_sum_all_Years.columns.difference(['siteid'])
df_sum_all_Years['Distinct_years'] = df_sum_all_Years[cols].gt(0).sum(axis=1)
df_sum_all_Years.to_csv("sum all years.csv")
print(df_sum_all_Years)
print ("num of all sites: ", len(df_sum_all_Years))


less_8 =df_sum_all_Years[df_sum_all_Years['Distinct_years'] !=8]
less_8.to_csv("7 years.csv")
print("num of sites with less years: ", len(less_8))
import numpy as np
x = np.array(less_8['Distinct_years'])
print(np.unique(x))