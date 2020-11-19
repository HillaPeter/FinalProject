import tableone
import pandas as pd
from tableone import TableOne

path = "/tmp/pycharm_project_355/"

# -------------read csv---------------------
df_2010_2011 = pd.read_csv("/mnt/nadavrap-students/STS/data/data_Shapira_20200911_2010_2011.csv")
df_2012_2013 = pd.read_csv("/mnt/nadavrap-students/STS/data/data_Shapira_20200911_2012_2013.csv")
df_2014_2015 = pd.read_csv("/mnt/nadavrap-students/STS/data/data_Shapira_20200911_2014_2015.csv")
df_2016_2017 = pd.read_csv("/mnt/nadavrap-students/STS/data/data_Shapira_20200911_2016_2017.csv")
df_2018_2019 = pd.read_csv("/mnt/nadavrap-students/STS/data/data_Shapira_20200911_2018_2019.csv")


mytable_2010_2011 = TableOne(data=df_2010_2011)
mytable_2012_2013 = TableOne(data=df_2012_2013)
mytable_2014_2015 = TableOne(data=df_2014_2015)
mytable_2016_2017 = TableOne(data=df_2016_2017)
mytable_2018_2019 = TableOne(data=df_2018_2019)

mytable_2010_2011.to_csv("2010_2011_tables.csv")
mytable_2012_2013.to_csv("2012_2013_tables.csv")
mytable_2014_2015.to_csv("2014_2015_tables.csv")
mytable_2016_2017.to_csv("2016_2017_tables.csv")
mytable_2018_2019.to_csv("2018_2019_tables.csv")

