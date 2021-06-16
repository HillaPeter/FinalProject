import pandas as pd
import matplotlib.pyplot as plt

path="/tmp/pycharm_project_355/"

    #-------------read csv---------------------
df_2010_2011 = pd.read_csv("/mnt/nadavrap-students/STS/data/data_Shapira_20200911_2010_2011.csv")
df_2012_2013 = pd.read_csv("/mnt/nadavrap-students/STS/data/data_Shapira_20200911_2012_2013.csv")
df_2014_2015 = pd.read_csv("/mnt/nadavrap-students/STS/data/data_Shapira_20200911_2014_2015.csv")
df_2016_2017 = pd.read_csv("/mnt/nadavrap-students/STS/data/data_Shapira_20200911_2016_2017.csv")
df_2018_2019 = pd.read_csv("/mnt/nadavrap-students/STS/data/data_Shapira_20200911_2018_2019.csv")

df_2010_2011.fillna(0)
df_2012_2013.fillna(0)
df_2014_2015.fillna(0)
df_2016_2017.fillna(0)
df_2018_2019.fillna(0)

# def merge_all_data():
#     df_1 =pd.concat([df_2010_2011,df_2012_2013],ignore_index=True)
#     df_1.to_csv(path+"somedata.csv")
#     # df_2 =pd.concat([df_1,df_2014_2015],ignore_index=True)
#     # df_3 =pd.concat([df_2,df_2016_2017],ignore_index=True)
#     # df_4 =pd.concat([df_3,df_2018_2019],ignore_index=True)
#     # df_4.to_csv(path+"allData.csv")
#     return df_1

# def group_by_years(column_name):
#     df_2010 = df_2010_2011.groupby('siteid')[column_name].apply(lambda x: (x== 2010 ).sum()).reset_index(name='2010')
#     df_2011 = df_2010_2011.groupby('siteid')[column_name].apply(lambda x: (x== 2011 ).sum()).reset_index(name='2011')
#     df_2012 = df_2012_2013.groupby('siteid')[column_name].apply(lambda x: (x== 2012 ).sum()).reset_index(name='2012')
#     df_2013 = df_2012_2013.groupby('siteid')[column_name].apply(lambda x: (x== 2013 ).sum()).reset_index(name='2013')
#     df_2014 = df_2014_2015.groupby('siteid')[column_name].apply(lambda x: (x== 2014 ).sum()).reset_index(name='2014')
#     df_2015 = df_2014_2015.groupby('siteid')[column_name].apply(lambda x: (x== 2015 ).sum()).reset_index(name='2015')
#     df_2016 = df_2016_2017.groupby('siteid')[column_name].apply(lambda x: (x== 2016 ).sum()).reset_index(name='2016')
#     df_2017 = df_2016_2017.groupby('siteid')[column_name].apply(lambda x: (x== 2017 ).sum()).reset_index(name='2017')
#     df_2018 = df_2018_2019.groupby('siteid')[column_name].apply(lambda x: (x== 2018 ).sum()).reset_index(name='2018')
#     df_2019 = df_2018_2019.groupby('siteid')[column_name].apply(lambda x: (x== 2019 ).sum()).reset_index(name='2019')
#
#     df1 = pd.merge(df_2010, df_2011, on='siteid')
#     df2 = pd.merge(df1, df_2012, on='siteid')
#     df3 = pd.merge(df2, df_2013, on='siteid')
#     df4 = pd.merge(df3, df_2014, on='siteid')
#     df5 = pd.merge(df4, df_2015, on='siteid')
#     df6 = pd.merge(df5, df_2016, on='siteid')
#     df7 = pd.merge(df6, df_2017, on='siteid')
#     df8 = pd.merge(df7, df_2018, on='siteid')
#     df_all = pd.merge(df8, df_2019, on='siteid')
#
#     return df_all

def group_by_sum(group_by_value,column_name,lambda_val):
    df_2010_2011_gb = df_2010_2011.groupby(group_by_value)[column_name].apply(lambda x: (x== lambda_val ).sum()).reset_index(name=column_name)
    df_2012_2013_gb = df_2012_2013.groupby(group_by_value)[column_name].apply(lambda x: (x== lambda_val ).sum()).reset_index(name=column_name)
    df_2014_2015_gb = df_2014_2015.groupby(group_by_value)[column_name].apply(lambda x: (x== lambda_val ).sum()).reset_index(name=column_name)
    df_2016_2017_gb = df_2016_2017.groupby(group_by_value)[column_name].apply(lambda x: (x== lambda_val ).sum()).reset_index(name=column_name)
    df_2018_2019_gb = df_2018_2019.groupby(group_by_value)[column_name].apply(lambda x: (x== lambda_val ).sum()).reset_index(name=column_name)

    df_merge_1=pd.merge(df_2010_2011_gb,df_2012_2013_gb, on=group_by_value)
    df_merge_2=pd.merge(df_merge_1,df_2014_2015_gb, on=group_by_value)
    df_merge_3=pd.merge(df_merge_2,df_2016_2017_gb, on=group_by_value)
    df_merge_4=pd.merge(df_merge_3,df_2018_2019_gb, on=group_by_value)

    cols = df_merge_4.columns.difference([group_by_value])
    df_merge_4[column_name] = df_merge_4.loc[:,cols].sum(axis=1)

    df_new=pd.DataFrame()
    df_new[group_by_value] = df_merge_4[group_by_value]
    df_new[column_name] = df_merge_4[column_name]

    return df_new

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

def group_by_mean(group_by_value,column_name,name):
    df_2010_2011_gb_sum = df_2010_2011.groupby(group_by_value)[column_name].sum().reset_index(name=name)
    df_2010_2011_gb_count = df_2010_2011.groupby(group_by_value)[column_name].count().reset_index(name=name)
    df_2012_2013_gb_sum = df_2012_2013.groupby(group_by_value)[column_name].sum().reset_index(name=name)
    df_2012_2013_gb_count = df_2012_2013.groupby(group_by_value)[column_name].count().reset_index(name=name)
    df_2014_2015_gb_sum = df_2014_2015.groupby(group_by_value)[column_name].sum().reset_index(name=name)
    df_2014_2015_gb_count = df_2014_2015.groupby(group_by_value)[column_name].count().reset_index(name=name)
    df_2016_2017_gb_sum = df_2016_2017.groupby(group_by_value)[column_name].sum().reset_index(name=name)
    df_2016_2017_gb_count = df_2016_2017.groupby(group_by_value)[column_name].count().reset_index(name=name)
    df_2018_2019_gb_sum = df_2018_2019.groupby(group_by_value)[column_name].sum().reset_index(name=name)
    df_2018_2019_gb_count = df_2018_2019.groupby(group_by_value)[column_name].count().reset_index(name=name)

    df_merge_1_sum=pd.merge(df_2010_2011_gb_sum,df_2012_2013_gb_sum, on=group_by_value)
    df_merge_2_sum=pd.merge(df_merge_1_sum,df_2014_2015_gb_sum, on=group_by_value)
    df_merge_3_sum=pd.merge(df_merge_2_sum,df_2016_2017_gb_sum, on=group_by_value)
    df_merge_4_sum=pd.merge(df_merge_3_sum,df_2018_2019_gb_sum, on=group_by_value)

    df_merge_1_count = pd.merge(df_2010_2011_gb_count, df_2012_2013_gb_count, on=group_by_value)
    df_merge_2_count = pd.merge(df_merge_1_count, df_2014_2015_gb_count, on=group_by_value)
    df_merge_3_count = pd.merge(df_merge_2_count, df_2016_2017_gb_count, on=group_by_value)
    df_merge_4_count = pd.merge(df_merge_3_count, df_2018_2019_gb_count, on=group_by_value)


    cols_sum = df_merge_4_sum.columns.difference([group_by_value])
    df_merge_4_sum[name] = df_merge_4_sum.loc[:,cols_sum].sum(axis=1)

    cols_count = df_merge_4_count.columns.difference([group_by_value])
    df_merge_4_count[name] = df_merge_4_count.loc[:, cols_count].sum(axis=1)

    df_new=pd.DataFrame()
    df_new[group_by_value] = df_merge_4_sum[group_by_value]
    df_new[name] = df_merge_4_sum[name]/df_merge_4_count[name]

    return df_new

def group_by_sum_2_values(group_by_value,column_name,lambda_val_start,lambda_val_end):
    df_2010_2011_gb = df_2010_2011.groupby(group_by_value)[column_name].apply(lambda x: ((x>= lambda_val_start) & (x<lambda_val_end)).sum()).reset_index(name=column_name)
    df_2012_2013_gb = df_2012_2013.groupby(group_by_value)[column_name].apply(lambda x: ((x>= lambda_val_start) & (x<lambda_val_end)).sum()).reset_index(name=column_name)
    df_2014_2015_gb = df_2014_2015.groupby(group_by_value)[column_name].apply(lambda x: ((x>= lambda_val_start) & (x<lambda_val_end)).sum()).reset_index(name=column_name)
    df_2016_2017_gb = df_2016_2017.groupby(group_by_value)[column_name].apply(lambda x: ((x>= lambda_val_start) & (x<lambda_val_end)).sum()).reset_index(name=column_name)
    df_2018_2019_gb = df_2018_2019.groupby(group_by_value)[column_name].apply(lambda x: ((x>= lambda_val_start) & (x<lambda_val_end)).sum()).reset_index(name=column_name)

    df_merge_1=pd.merge(df_2010_2011_gb,df_2012_2013_gb, on=group_by_value)
    df_merge_2=pd.merge(df_merge_1,df_2014_2015_gb, on=group_by_value)
    df_merge_3=pd.merge(df_merge_2,df_2016_2017_gb, on=group_by_value)
    df_merge_4=pd.merge(df_merge_3,df_2018_2019_gb, on=group_by_value)

    cols = df_merge_4.columns.difference([group_by_value])
    df_merge_4[column_name] = df_merge_4.loc[:,cols].sum(axis=1)

    df_new=pd.DataFrame()
    df_new[group_by_value] = df_merge_4[group_by_value]
    df_new[column_name] = df_merge_4[column_name]

    return df_new

values=['siteid','surgid','hospid']

for val in values:
    df_mortality= group_by_sum(val,'mt30stat',2)
    df_count_opr=group_by_count(val,'total')
    df_merge_1=pd.merge(df_mortality,df_count_opr,on=val)

    df_age=group_by_mean(val,'age','mean_age')
    df_merge_2=pd.merge(df_merge_1,df_age,on=val)

    df_fhcad=group_by_sum(val,'fhcad',1)
    df_merge_3=pd.merge(df_merge_2,df_fhcad,on=val)

    df_weightkg=group_by_mean(val,'weightkg','mean_weightkg')
    df_merge_4=pd.merge(df_merge_3,df_weightkg,on=val)

    df_diabetes=group_by_sum(val,'diabetes',1)
    df_merge_5=pd.merge(df_merge_4,df_diabetes,on=val)

    df_predmort=group_by_mean(val,'predmort','mean_predmort')
    df_merge_6=pd.merge(df_merge_5,df_predmort,on=val)

    df_predreop=group_by_mean(val,'predreop','mean_predreop')
    df_merge_7=pd.merge(df_merge_6,df_predreop,on=val)

    df_incidenc=group_by_sum_2_values(val,'incidenc',2,6)
    df_merge_8=pd.merge(df_merge_7,df_incidenc,on=val)

    df_tobaccouse=group_by_sum_2_values(val,'tobaccouse',2,6)
    df_merge_9=pd.merge(df_merge_8,df_tobaccouse,on=val)

    df_chrlungd=group_by_sum_2_values(val,'chrlungd',2,6)
    df_merge_10=pd.merge(df_merge_9,df_chrlungd,on=val)

    df_merge_10['precent_mortality/total']=(df_merge_10['mt30stat']/df_merge_10['total'])*100
    df_merge_10['precent_reop/total']=(df_merge_10['incidenc']/df_merge_10['total'])*100

    df_cancer=group_by_sum(val,'cancer',1)
    df_merge_11=pd.merge(df_merge_10,df_cancer,on=val)

    df_PVD=group_by_sum(val,'pvd',1)
    df_merge_12=pd.merge(df_merge_11,df_PVD,on=val)

    df_liverdis=group_by_sum(val,'liverdis',1)
    df_merge_13=pd.merge(df_merge_12,df_liverdis,on=val)

    df_alcohol=group_by_sum(val,'alcohol',1)
    df_merge_14=pd.merge(df_merge_13,df_alcohol,on=val)

    df_female = group_by_sum(val,'gender', 2)
    df_merge_15 = pd.merge(df_merge_14, df_female, on=val)

    df_male = group_by_sum(val, 'gender', 1)
    df_merge_16 = pd.merge(df_merge_15, df_male, on=val)

    df_merge_16['female']=df_merge_16['gender_x']
    df_merge_16['male']=df_merge_16['gender_y']
    df_merge_16.drop(['gender_x', 'gender_y'], inplace=True, axis=1)

    df_merge_16.to_csv(path+val+"_new_data.csv")

#surgid
