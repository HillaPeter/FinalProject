import pandas as pd
from tableone import TableOne
df_map=pd.read_csv("/mnt/nadavrap-students/STS/data/imputed_data2.csv")

# cols=df_map.columns
# for i in cols:
#     print(i)
# print(df_map['Reoperation'].to_dict())

#Reoperation- First Time-0, Reoperation-1
#gender- male -0 female - 1
#Ethnicity Non-Hispanic-0 Hispanic-1
#prcab- Previous Non-CAB-0 Previous CAB-
# VDInsufA,VDInsufM None/Trivial/Trace/Mild-0 Moderate/Severe-1
#Status Elective- 0  Urgent-1
#Mt30Stat Unknown-0 Alive-1
#Incidenc NA - Not a cardiovascular surgery-0 First cardiovascular surgery-0 First re-op cardiovascular surgery-0 "Second re-op cardiovascular surgery":1,"Third re-op cardiovascular surgery":1,"Fourth or more re-op cardiovascular surgery":1
#Reoperation First Time-0 Reoperation-1
#SmokingStatus Never smoker-0 Smoker-1
#ClassNYHGroup I/II-0 III/IV-1
#NumDisV_ordinal One- 1 Two-1 Three-1
#NumRadDA

# df_map = df_map.drop(columns=['SiteID', 'surgyear','surgid'], axis=1)
#
# first=df_map.loc[df_map['Reoperation']=='First Time']
# reop=df_map.loc[df_map['Reoperation']=='Reoperation']
#
# columns=first.columns
# def update(df,title):
#     # categories=[]
#     for col in columns:
#         try:
#             df = df.replace({col: {False: 0, True: 1}})
#             df = df.replace({col: {"No": 0, "Yes": 1}})
#             df = df.replace({col: {"Male": 0, "Female": 1}})
#             df = df.replace({col: {"Elective": 0, "Urgent": 1}})
#             df = df.replace({col: {"Non-Hispanic": 0, "Hispanic": 1}})
#             df = df.replace({col: {"Previous Non-CAB": 0, "Previous CAB": 1}})
#             df = df.replace({col: {"None/Trivial/Trace/Mild": 0, "Moderate/Severe": 1}})
#             df = df.replace({col: {"Unknown": 1, "Alive": 1, "Dead": 0}})
#             df = df.replace({col: {"First cardiovascular surgery": 0, "NA - Not a cardiovascular surgery":0,"First re-op cardiovascular surgery": 0,"Second re-op cardiovascular surgery":1,"Third re-op cardiovascular surgery":1,"Fourth or more re-op cardiovascular surgery":1}})
#             # df = df.replace({col: {"First Time": 0, "Reoperation": 1}})
#             df = df.replace({col: {"Never smoker": 0, "Smoker": 1}})
#             df = df.replace({col: {"I/II": 0, "III/IV": 1}})
#             df = df.replace({col: {"None":0,"One": 1,"Two": 1, "Three": 1}})
#             df = df.replace({col: {int("2"):1,int("3"): 1,int("4"): 1, int("5"): 1, int("6"): 1}})
#             # categories.append(col)
#
#         except:
#             print(col)
#
#
#     df.to_csv(title+".csv")
#
# update(first,"op_first")
# update(reop,"op_reop")

###########################################3

# def createtable(df,op):
#
#     df.rename(columns={'Unnamed: 0': 'number'}, inplace=True)
#     print(df.columns)
#     df=df.drop(columns=['number'], axis=1)
#     df=df.drop(columns=['HospID'], axis=1)
#     columns=df.columns
#
#     remove_cols=['PLOS','Age','CreatLst','HDEF','PerfusTm','XClampTm','DistVein','VentHrsTot','PredMort','PredDeep','PredReop',
#                  'PredStro','PredVent','PredRenF','PredMM','Pred6D','Pred14D','ICUHrsTotal','BMI','NumIMADA','NumRadDA','TotalNumberOfGrafts']
#
#     categorical_vals=[]
#     for element in columns:
#         if element not in remove_cols:
#             categorical_vals.append(element)
#     mytable= TableOne(data=df, categorical=categorical_vals)
#     mytable.to_csv(op+"_table.csv")
#
# df=pd.read_csv("op_first.csv")
# df1=pd.read_csv("op_reop.csv")
# createtable(df,"op_first_table")
# createtable(df1,"op_reop_table")


############################################################################
##merge

# df=pd.read_csv("op_first_table_table.csv")
# df1=pd.read_csv("op_reop_table_table.csv")
#
# df = df.rename(columns={'Unnamed: 0': 'values', 'Unnamed: 1':'type','Missing':'Missing_first','Overall':'first' })
# df1 = df1.rename(columns={'Unnamed: 0': 'values', 'Unnamed: 1':'type','Missing':'Missing_reoperation','Overall':'reoperation'})
#
# merge=pd.merge(df,df1,on=["values","type"],how="inner")
#
# merge.to_excel("merge.xls")


#########################################################################

df_all=pd.read_excel("merge.xls")
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
            if str(df[line][2]) == '0.0':
                df[line][2] = 'Dead'
            if str(df[line][2]) == '1.0':
                to_remove.append(df[line][0])
        elif value == "Mt30Stat" or value=="Mortalty":
            to_remove.append(df[line][0])
        elif value == "MtOpD":
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
            elif str(df[line][2]) == '2.0' or str(df[line][2]) == '3.0' or str(df[line][2]) == '3' or str(df[line][2]) == '2':
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
savedf.to_excel("tableone.xls")