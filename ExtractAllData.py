import pandas as pd
from tableone import TableOne
df=pd.read_csv("/mnt/nadavrap-students/STS/data/imputed_data2.csv")

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

df = df.drop(columns=['SiteID', 'surgyear','surgid','Mt30Stat','Mortalty'], axis=1)

for col in df.columns:
    try:
        df = df.replace({col: {False: 0, True: 1}})
        df = df.replace({col: {"No": 0, "Yes": 1}})
        df = df.replace({col: {"Male": 0, "Female": 1}})
        df = df.replace({col: {"Elective": 0, "Urgent": 1}})
        df = df.replace({col: {"Non-Hispanic": 0, "Hispanic": 1}})
        df = df.replace({col: {"Previous Non-CAB": 0, "Previous CAB": 1}})
        df = df.replace({col: {"None/Trivial/Trace/Mild": 0, "Moderate/Severe": 1}})
        df = df.replace({col: {"Unknown": 1, "Alive": 1, "Dead": 0}})
        df = df.replace({col: {"First cardiovascular surgery": 0, "NA - Not a cardiovascular surgery":0,"First re-op cardiovascular surgery": 0,"Second re-op cardiovascular surgery":1,"Third re-op cardiovascular surgery":1,"Fourth or more re-op cardiovascular surgery":1}})
        df = df.replace({col: {"First Time": 0, "Reoperation": 1}})
        df = df.replace({col: {"Never smoker": 0, "Smoker": 1}})
        df = df.replace({col: {"I/II": 0, "III/IV": 1}})
        df = df.replace({col: {"None":0,"One": 1,"Two": 1, "Three": 1}})
        df = df.replace({col: {int("2"):1,int("3"): 1,int("4"): 1, int("5"): 1, int("6"): 1}})
        # categories.append(col)

    except:
        x="None"#print(col)


df.to_csv("clean_data_tableone.csv")


###########################################3

df=pd.read_csv("clean_data_tableone.csv")

df.rename(columns={'Unnamed: 0': 'number'}, inplace=True)
print("before ",len(df.columns))
df=df.drop(columns=['number'], axis=1)
df=df.drop(columns=['HospID'], axis=1)
# df=df.drop(columns=['Mortalty'], axis=1)
# df=df.drop(columns=['Mt30Stat'], axis=1)

columns=df.columns
print("After ",len(df.columns))
remove_cols=['PLOS','Age','CreatLst','HDEF','PerfusTm','XClampTm','DistVein','VentHrsTot','PredMort','PredDeep','PredReop',
             'PredStro','PredVent','PredRenF','PredMM','Pred6D','Pred14D','ICUHrsTotal','BMI','NumIMADA','NumRadDA','TotalNumberOfGrafts']
print("---")

list_columns=[]
for column in columns:
    list_columns.append(column)
print(list_columns)

groupby='Reoperation'

categorical=[]
for element in columns:
    if element not in remove_cols:
        categorical.append(element)

mytable= TableOne(data=df,columns=list_columns,categorical=categorical,groupby=groupby,pval=True,smd=True,htest_name=True)#,min_max=remove_cols)
#data, columns, categorical, groupby, nonnormal, pval = True, smd=True,htest_name=True
mytable.to_csv("_table.csv")





############################################################################
##merge

df=pd.read_csv("op_first_table_table.csv")
df1=pd.read_csv("op_reop_table_table.csv")

df = df.rename(columns={'Unnamed: 0': 'values', 'Unnamed: 1':'type','Missing':'Missing_first','Overall':'first' })
df1 = df1.rename(columns={'Unnamed: 0': 'values', 'Unnamed: 1':'type','Missing':'Missing_reoperation','Overall':'reoperation'})

merge=pd.merge(df,df1,on=["values","type"],how="inner")

merge.to_excel("merge.xls")


#########################################################################

df_all=pd.read_csv("_table.csv")

df_all.reset_index()

df_all.to_csv("_table1.csv")
#########################################################################
df_all=pd.read_csv("_table1.csv")
print(df_all.columns)
df_all = df_all.rename(columns={'Unnamed: 0': 'index','Unnamed: 0.1': 'value', 'Unnamed: 1':'type'})
print(df_all.columns)
df = df_all.T

to_remove=[]
for line in df:
        # value=df[line][1]
        # print(value)
        value = df[line][1]
        # print(value)
        index = int(df[line][0])
        value = str(value)
        type=df[line][2]
        if value=="Gender":
            if str(type)=='1.0':
                df[line][2]='Female'
            if str(type)=='0.0':
                to_remove.append(index)
        elif value=="Ethnicity":
            if str(type)=='1.0':
                df[line][2] = 'Hispanic'
            if str(type)=='0.0':
                to_remove.append(index)
        elif value=="prcab":
            if str(type)=='1.0':
                df[line][2] = 'Previous CAB'
            if str(type)=='0.0':
                to_remove.append(index)
        elif value == "VDInsufA" or value=="VDInsufM":
            if str(type) == '1.0':
                df[line][2] = 'Moderate/Severe'
            if str(type) == '0.0':
                to_remove.append(index)
        elif value == "Status":
            if str(type) == '0.0':
                to_remove.append(index)
            if str(type) == '1.0':
                df[line][2] = 'Alive'
        elif value == "MtOpD":
            if str(type) == '1.0':
                df[line][2] = 'Alive'
            if str(type) == '0.0':
                to_remove.append(index)
        elif value == "Incidenc":
            if str(type) == '1.0':
                df[line][2] = 'First re-op cardiovascular surgery or more'
            if str(type) == '0.0':
                to_remove.append(index)
        # elif value == "Reoperation":
        #     if str(df[line][2]) == '1.0':
        #         df[line][2] = 'Reoperation'
        #     if str(df[line][2]) == '0.0':
        #         to_remove.append(df[line][0])
        elif value == "SmokingStatus":
            if str(type) == '1.0':
                df[line][2] = 'Smoker'
            if str(type) == '0.0':
                to_remove.append(index)
        elif value == "ClassNYHGroup":
            if str(type) == '0.0':
                df[line][2] = 'I/II'
            if str(type) == '1.0':
                to_remove.append(index)
        elif value == "NumDisV_ordinal":
            if str(type) == '0.0':
                df[line][2] = 'None'
            if str(type) == '1.0':
                to_remove.append(index)
        elif value == "NumDisV":
            if str(type) == '1.0':
                df[line][2] = 'None'
            elif str(type) == '2.0' or str(type) == '3.0' or str(type) == '3' or str(type) == '2':
                to_remove.append(index)
        elif value!=None and value!="nan" and value!="n":
            # print(value)
            value=value[:-7]
            if str(type)=='0.0':
                to_remove.append(index)
            if str(type)=='1.0':
                df[line][2] = 'Yes'


print(df)
print(to_remove)
savedf=df.T

# for remove in to_remove:
savedf.drop(savedf.index[to_remove], inplace=True)
# savedf.drop('Unnamed: 0', inplace=True)
savedf.to_excel("tableone.xls")