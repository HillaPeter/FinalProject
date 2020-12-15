import pandas as pd
from tableone import TableOne
# df_map=pd.read_csv("imputed_data2.csv")
#
# #surgyear
# df_2010=df_map.loc[df_map['surgyear']==2010]
# df_2011=df_map.loc[df_map['surgyear']==2011]
# df_2012=df_map.loc[df_map['surgyear']==2012]
# df_2013=df_map.loc[df_map['surgyear']==2013]
# df_2014=df_map.loc[df_map['surgyear']==2014]
# df_2015=df_map.loc[df_map['surgyear']==2015]
# df_2016=df_map.loc[df_map['surgyear']==2016]
# df_2017=df_map.loc[df_map['surgyear']==2017]
# df_2018=df_map.loc[df_map['surgyear']==2018]
# df_2019=df_map.loc[df_map['surgyear']==2019]
#
#
# #gender- male -0 female - 1
# #Ethnicity Non-Hispanic-0 Hispanic-1
# #prcab- Previous Non-CAB-0 Previous CAB-
# # VDInsufA,VDInsufM None/Trivial/Trace/Mild-0 Moderate/Severe-1
# #Status Elective- 0  Urgent-1
# #Mt30Stat Unknown-0 Alive-1
# #Incidenc NA - Not a cardiovascular surgery-0 First cardiovascular surgery-0 First re-op cardiovascular surgery-0 "Second re-op cardiovascular surgery":1,"Third re-op cardiovascular surgery":1,"Fourth or more re-op cardiovascular surgery":1
# #Reoperation First Time-0 Reoperation-1
# #SmokingStatus Never smoker-0 Smoker-1
# #ClassNYHGroup I/II-0 III/IV-1
# #NumDisV_ordinal One- 1 Two-1 Three-1
# #NumRadDA
# columns=df_2010.columns
# def update(df,year):
#     categories=[]
#     for col in columns:
#         try:
#             df = df.replace({col: {False: 0, True: 1}})
#             df = df.replace({col: {"No": 0, "Yes": 1}})
#             df = df.replace({col: {"Male": 0, "Female": 1}})
#             df = df.replace({col: {"Elective": 0, "Urgent": 1}})
#             df = df.replace({col: {"Non-Hispanic": 0, "Hispanic": 1}})
#             df = df.replace({col: {"Previous Non-CAB": 0, "Previous CAB": 1}})
#             df = df.replace({col: {"None/Trivial/Trace/Mild": 0, "Moderate/Severe": 1}})
#             df = df.replace({col: {"Unknown": 0, "Alive": 1, "Dead": 0}})
#             df = df.replace({col: {"First cardiovascular surgery": 0, "NA - Not a cardiovascular surgery":0,"First re-op cardiovascular surgery": 0,"Second re-op cardiovascular surgery":1,"Third re-op cardiovascular surgery":1,"Fourth or more re-op cardiovascular surgery":1}})
#             df = df.replace({col: {"First Time": 0, "Reoperation": 1}})
#             df = df.replace({col: {"Never smoker": 0, "Smoker": 1}})
#             df = df.replace({col: {"I/II": 0, "III/IV": 1}})
#             df = df.replace({col: {"None":0,"One": 1,"Two": 1, "Three": 1}})
#             df = df.replace({col: {int("2"):1,int("3"): 1,int("4"): 1, int("5"): 1, int("6"): 1}})
#             categories.append(col)
#
#         except:
#             print(col)
#
#
#     df.to_csv("new_"+year+".csv")
#
# update(df_2010,"2010")
# update(df_2011,"2011")
# update(df_2012,"2012")
# update(df_2013,"2013")
# update(df_2014,"2014")
# update(df_2015,"2015")
# update(df_2016,"2016")
# update(df_2017,"2017")
# update(df_2018,"2018")
# update(df_2019,"2019")
###########################################3
# #
# def createtable(df,year):
#
#
#     # df_2010.drop(df_2010[], axis=1)
#     df=df.drop(columns=['SiteID','surgyear'], axis=1)
#     columns=df.columns
#
#     # print("1 ", len(columns))
#
#     remove_cols=['PLOS','Age','CreatLst','HDEF','PerfusTm','XClampTm','DistVein','VentHrsTot','PredMort','PredDeep','PredReop',
#                  'PredStro','PredVent','PredRenF','PredMM','Pred6D','Pred14D','ICUHrsTotal','BMI','NumIMADA','NumRadDA','TotalNumberOfGrafts']
#
#     categorical_vals=[]
#     for element in columns:
#         if element not in remove_cols:
#             categorical_vals.append(element)
#     # print("3 ", len(categorical_vals))
#     mytable_2010 = TableOne(data=df, categorical=categorical_vals)
#     mytable_2010.to_csv(year+"_table.csv")
#
# df=pd.read_csv("new_2010.csv")
# df1=pd.read_csv("new_2011.csv")
# df2=pd.read_csv("new_2012.csv")
# df3=pd.read_csv("new_2013.csv")
# df4=pd.read_csv("new_2014.csv")
# df5=pd.read_csv("new_2015.csv")
# df6=pd.read_csv("new_2016.csv")
# df7=pd.read_csv("new_2017.csv")
# df8=pd.read_csv("new_2018.csv")
# df9=pd.read_csv("new_2019.csv")
#
# createtable(df,"2010")
# createtable(df1,"2011")
# createtable(df2,"2012")
# createtable(df3,"2013")
# createtable(df4,"2014")
# createtable(df5,"2015")
# createtable(df6,"2016")
# createtable(df7,"2017")
# createtable(df8,"2018")
# createtable(df9,"2019")

############################################################################
#merge
#
#
# df_2010=pd.read_csv("2010_table.csv")
# df_2011=pd.read_csv("2011_table.csv")
# df_2012=pd.read_csv("2012_table.csv")
# df_2013=pd.read_csv("2013_table.csv")
# df_2014=pd.read_csv("2014_table.csv")
# df_2015=pd.read_csv("2015_table.csv")
# df_2016=pd.read_csv("2016_table.csv")
# df_2017=pd.read_csv("2017_table.csv")
# df_2018=pd.read_csv("2018_table.csv")
# df_2019=pd.read_csv("2019_table.csv")
#
# # print(df_2010)
# df_2010 = df_2010.rename(columns={'Unnamed: 0': 'values', 'Unnamed: 1':'type','Missing':'Missing_2010','Overall':'Overall_2010' })
# df_2011 = df_2011.rename(columns={'Unnamed: 0': 'values', 'Unnamed: 1':'type','Missing':'Missing_2011','Overall':'Overall_2011'})
# df_2012 = df_2012.rename(columns={'Unnamed: 0': 'values', 'Unnamed: 1':'type','Missing':'Missing_2012','Overall':'Overall_2012'})
# df_2013 = df_2013.rename(columns={'Unnamed: 0': 'values', 'Unnamed: 1':'type','Missing':'Missing_2013','Overall':'Overall_2013'})
# df_2014 = df_2014.rename(columns={'Unnamed: 0': 'values', 'Unnamed: 1':'type','Missing':'Missing_2014','Overall':'Overall_2014'})
# df_2015 = df_2015.rename(columns={'Unnamed: 0': 'values', 'Unnamed: 1':'type','Missing':'Missing_2015','Overall':'Overall_2015'})
# df_2016 = df_2016.rename(columns={'Unnamed: 0': 'values', 'Unnamed: 1':'type','Missing':'Missing_2016','Overall':'Overall_2016'})
# df_2017 = df_2017.rename(columns={'Unnamed: 0': 'values', 'Unnamed: 1':'type','Missing':'Missing_2017','Overall':'Overall_2017'})
# df_2018 = df_2018.rename(columns={'Unnamed: 0': 'values', 'Unnamed: 1':'type','Missing':'Missing_2018','Overall':'Overall_2018'})
# df_2019 = df_2019.rename(columns={'Unnamed: 0': 'values', 'Unnamed: 1':'type','Missing':'Missing_2019','Overall':'Overall_2019'})
#
# df_0=pd.DataFrame(df_2018)
#
# df=pd.merge(df_0,df_2019,on=["values","type"],how="inner")
# print(df)
# df_1=pd.merge(df_2017,df,on=["values","type"],how='inner')
# df_2=pd.merge(df_2016,df_1,on=["values","type"],how='inner')
# df_3=pd.merge(df_2015,df_2,on=["values","type"],how='inner')
# df_4=pd.merge(df_2014,df_3,on=["values","type"],how='inner')
# df_5=pd.merge(df_2013,df_4,on=["values","type"],how='inner')
# df_6=pd.merge(df_2012,df_5,on=["values","type"],how='inner')
# df_7=pd.merge(df_2011,df_6,on=["values","type"],how='inner')
# df_8=pd.merge(df_2010,df_7,on=["values","type"],how='inner')
#
#
# print(df_8)
# df_8.to_excel("a.xls")


#########################################################################

df_all=pd.read_excel("a.xls")
df = df_all.T

to_remove=[]
for line in df:
        value=df[line][1]
        # print(value)
        index=df[line][0]
        value=str(value)
        if value=="Gender":
                if str(df[line][2])=='1.0':
                        df[line][2]='Female'
                if str(df[line][2])=='0.0':
                        to_remove.append(df[line][0])
        elif value=="Ethnicity":
                if str(df[line][2])=='1.0':
                        df[line][2]='Hispanic'
                if str(df[line][2])=='0.0':
                        to_remove.append(df[line][0])
        elif value=="prcab":
                if str(df[line][2])=='1.0':
                        df[line][2]='Previous CAB'
                if str(df[line][2])=='0.0':
                        to_remove.append(df[line][0])
        elif value == "VDInsufA" or value=="VDInsufM":
            if str(df[line][2]) == '1.0':
                df[line][2] = 'Moderate/Severe'
            if str(df[line][2]) == '0.0':
                to_remove.append(df[line][0])
        elif value == "Status":
            if str(df[line][2]) == '1.0':
                df[line][2] = 'Urgent'
            if str(df[line][2]) == '0.0':
                to_remove.append(df[line][0])
        elif value == "Mt30Stat":
            if str(df[line][2]) == '1.0':
                df[line][2] = 'Alive'
            if str(df[line][2]) == '0.0':
                to_remove.append(df[line][0])
        elif value == "Incidenc":
            if str(df[line][2]) == '1.0':
                df[line][2] = 'First re-op cardiovascular surgery'
            if str(df[line][2]) == '0.0':
                to_remove.append(df[line][0])
        elif value == "Reoperation":
            if str(df[line][2]) == '1.0':
                df[line][2] = 'Reoperation'
            if str(df[line][2]) == '0.0':
                to_remove.append(df[line][0])
        elif value == "SmokingStatus":
            if str(df[line][2]) == '1.0':
                df[line][2] = 'Smoker'
            if str(df[line][2]) == '0.0':
                to_remove.append(df[line][0])
        elif value == "ClassNYHGroup":
            if str(df[line][2]) == '0.0':
                df[line][2] = 'I/II'
            if str(df[line][2]) == '1.0':
                to_remove.append(df[line][0])
        elif value == "NumDisV_ordinal":
            if str(df[line][2]) == '0.0':
                df[line][2] = 'Two'
            if str(df[line][2]) == '1.0':
                to_remove.append(df[line][0])
        elif value == "NumDisV":
            if str(df[line][2]) == '1.0':
                df[line][2] = 'None'
            elif str(df[line][2]) == '2.0' or str(df[line][2]) == '3.0':
                to_remove.append(df[line][0])
        elif value!=None and value!="nan" and value!="n":
            print(value)
            value=value[:-7]
            if str(df[line][2])=='1.0':
                    df[line][2]='Yes'
            if str(df[line][2])=='0.0':
                    to_remove.append(df[line][0])

print(df)
savedf=df.T

# for remove in to_remove:
savedf.drop(savedf.index[to_remove], inplace=True)
# savedf.drop('Unnamed: 0', inplace=True)
savedf.to_excel("b.xls")