import tableone
import pandas as pd
from tableone import TableOne
import numpy as np

path = "/tmp/pycharm_project_355/"



#seperated years
# df_2010_2011 = pd.read_csv("/mnt/nadavrap-students/STS/data/data_Shapira_20200911_2010_2011.csv")
# df_2012_2013 = pd.read_csv("/mnt/nadavrap-students/STS/data/data_Shapira_20200911_2012_2013.csv")
# df_2014_2015 = pd.read_csv("/mnt/nadavrap-students/STS/data/data_Shapira_20200911_2014_2015.csv")
# df_2016_2017 = pd.read_csv("/mnt/nadavrap-students/STS/data/data_Shapira_20200911_2016_2017.csv")
# df_2018_2019 = pd.read_csv("/mnt/nadavrap-students/STS/data/data_Shapira_20200911_2018_2019.csv")
#
# mask = df_2010_2011['surgyear'] != 2010
# df_2011 = df_2010_2011[mask]
# df_2010 = df_2010_2011[~mask]
# mask2 = df_2012_2013['surgyear'] != 2012
# df_2013 = df_2012_2013[mask2]
# df_2012 = df_2012_2013[~mask2]
# mask3 = df_2014_2015['surgyear'] != 2014
# df_2015 = df_2014_2015[mask3]
# df_2014 = df_2014_2015[~mask3]
# mask4 = df_2016_2017['surgyear'] != 2016
# df_2017 = df_2016_2017[mask4]
# df_2016 = df_2016_2017[~mask4]
# mask5 = df_2018_2019['surgyear'] != 2018
# df_2019 = df_2018_2019[mask5]
# df_2018 = df_2018_2019[~mask5]
#
# df_2010.to_csv("2010.csv")
# df_2011.to_csv("2011.csv")
# df_2012.to_csv("2012.csv")
# df_2013.to_csv("2013.csv")
# df_2014.to_csv("2014.csv")
# df_2015.to_csv("2015.csv")
# df_2016.to_csv("2016.csv")
# df_2017.to_csv("2017.csv")
# df_2018.to_csv("2018.csv")
# df_2019.to_csv("2019.csv")

df_data=pd.read_csv("relevancedata.csv")
#
df = df_data.T
list_vals=[]
all_vals=[]
for value in df:
    categorial=(str(df[value][2]))
    if categorial=="Text (categorical values specified by User)":
        print(df[value][1], " un wanted")
    if categorial == "Text (categorical values specified by STS)":
        list_vals.append(str(df[value][1]))
        all_vals.append(str(df[value][1]))
    if categorial=="Integer" or categorial=="Real":
        all_vals.append(str(df[value][1]))

# print(list_vals)

# df_2010=pd.read_csv("2010.csv")
# df_2011=pd.read_csv("2011.csv")
# df_2012=pd.read_csv("2012.csv")
# df_2013=pd.read_csv("2013.csv")
# df_2014=pd.read_csv("2014.csv")
# df_2015=pd.read_csv("2015.csv")
# df_2016=pd.read_csv("2016.csv")
# df_2017=pd.read_csv("2017.csv")
# df_2018=pd.read_csv("2018.csv")
# df_2019=pd.read_csv("2019.csv")
#
# df_2010=pd.DataFrame(df_2010,columns=all_vals)
# df_2011=pd.DataFrame(df_2011,columns=all_vals)
# df_2012=pd.DataFrame(df_2012,columns=all_vals)
# df_2013=pd.DataFrame(df_2013,columns=all_vals)
# df_2014=pd.DataFrame(df_2014,columns=all_vals)
# df_2015=pd.DataFrame(df_2015,columns=all_vals)
# df_2016=pd.DataFrame(df_2016,columns=all_vals)
# df_2017=pd.DataFrame(df_2017,columns=all_vals)
# df_2018=pd.DataFrame(df_2018,columns=all_vals)
# df_2019=pd.DataFrame(df_2019,columns=all_vals)
#
#
# mytable_2010 = TableOne(data=df_2010,categorical=list_vals)
# mytable_2010.to_csv("2010_tables_1.csv")
# mytable_2011 = TableOne(data=df_2011,categorical=list_vals)
# mytable_2011.to_csv("2011_tables_1.csv")
# mytable_2012 = TableOne(data=df_2012,categorical=list_vals)
# mytable_2012.to_csv("2012_tables_1.csv")
# mytable_2013 = TableOne(data=df_2013,categorical=list_vals)
# mytable_2013.to_csv("2013_tables_1.csv")
# mytable_2014 = TableOne(data=df_2014,categorical=list_vals)
# mytable_2014.to_csv("2014_tables_1.csv")
# mytable_2015 = TableOne(data=df_2015,categorical=list_vals)
# mytable_2015.to_csv("2015_tables_1.csv")
# mytable_2016 = TableOne(data=df_2016,categorical=list_vals)
# mytable_2016.to_csv("2016_tables_1.csv")
# mytable_2017 = TableOne(data=df_2017,categorical=list_vals)
# mytable_2017.to_csv("2017_tables_1.csv")
# mytable_2018 = TableOne(data=df_2018,categorical=list_vals)
# mytable_2018.to_csv("2018_tables_1.csv")
# mytable_2019 = TableOne(data=df_2019,categorical=list_vals)
# mytable_2019.to_csv("2019_tables_1.csv")

df_2010=pd.read_csv("2010_tables.csv")
df_2011=pd.read_csv("2011_tables.csv")
df_2012=pd.read_csv("2012_tables.csv")
df_2013=pd.read_csv("2013_tables.csv")
df_2014=pd.read_csv("2014_tables.csv")
df_2015=pd.read_csv("2015_tables.csv")
df_2016=pd.read_csv("2016_tables.csv")
df_2017=pd.read_csv("2017_tables.csv")
df_2018=pd.read_csv("2018_tables.csv")
df_2019=pd.read_csv("2019_tables.csv")

# print(df_2010)
df_2010 = df_2010.rename(columns={'Unnamed: 0': 'values', 'Unnamed: 1':'type','Missing':'Missing_2010','Overall':'Overall_2010' })
df_2011 = df_2011.rename(columns={'Unnamed: 0': 'values', 'Unnamed: 1':'type','Missing':'Missing_2011','Overall':'Overall_2011'})
df_2012 = df_2012.rename(columns={'Unnamed: 0': 'values', 'Unnamed: 1':'type','Missing':'Missing_2012','Overall':'Overall_2012'})
df_2013 = df_2013.rename(columns={'Unnamed: 0': 'values', 'Unnamed: 1':'type','Missing':'Missing_2013','Overall':'Overall_2013'})
df_2014 = df_2014.rename(columns={'Unnamed: 0': 'values', 'Unnamed: 1':'type','Missing':'Missing_2014','Overall':'Overall_2014'})
df_2015 = df_2015.rename(columns={'Unnamed: 0': 'values', 'Unnamed: 1':'type','Missing':'Missing_2015','Overall':'Overall_2015'})
df_2016 = df_2016.rename(columns={'Unnamed: 0': 'values', 'Unnamed: 1':'type','Missing':'Missing_2016','Overall':'Overall_2016'})
df_2017 = df_2017.rename(columns={'Unnamed: 0': 'values', 'Unnamed: 1':'type','Missing':'Missing_2017','Overall':'Overall_2017'})
df_2018 = df_2018.rename(columns={'Unnamed: 0': 'values', 'Unnamed: 1':'type','Missing':'Missing_2018','Overall':'Overall_2018'})
df_2019 = df_2019.rename(columns={'Unnamed: 0': 'values', 'Unnamed: 1':'type','Missing':'Missing_2019','Overall':'Overall_2019'})

# print(df_2010)

df_0=pd.DataFrame(df_2018)
df=pd.merge(df_0,df_2019,on=["values","type"],how="outer")
df_1=pd.merge(df_2017,df,on=["values","type"],how='outer')
df_2=pd.merge(df_2016,df_1,on=["values","type"],how='outer')
df_3=pd.merge(df_2015,df_2,on=["values","type"],how='outer')
df_4=pd.merge(df_2014,df_3,on=["values","type"],how='outer')
df_5=pd.merge(df_2013,df_4,on=["values","type"],how='outer')
df_6=pd.merge(df_2012,df_5,on=["values","type"],how='outer')
df_7=pd.merge(df_2011,df_6,on=["values","type"],how='outer')
df_8=pd.merge(df_2010,df_7,on=["values","type"],how='outer')


print(df_8)
df_8.to_excel("2010-2019-only cayegorial.xls")