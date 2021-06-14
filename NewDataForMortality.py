import pandas as pd
from sklearn import linear_model
import statsmodels.api as sm
import numpy as np
from scipy import stats


# df_2018 = pd.read_csv("/mnt/nadavrap-students/STS/data/2018_2019.csv")
# df_2016 = pd.read_csv("/mnt/nadavrap-students/STS/data/2016_2017.csv")
# df_2014 = pd.read_csv("/mnt/nadavrap-students/STS/data/2014_2015.csv")
# df_2012 = pd.read_csv("/mnt/nadavrap-students/STS/data/2012_2013.csv")
# df_2010 = pd.read_csv("/mnt/nadavrap-students/STS/data/2010_2011.csv")
#
# print (df_2018.stsrcom.unique())
# print (df_2016.stsrcom.unique())
# print (df_2014.stsrcom.unique())
# print (df_2012.stsrcom.unique())
# print (df_2010.stsrcom.unique())
# print (df_2018.stsrcHospD.unique())
# print (df_2016.stsrcHospD.unique())
# print (df_2014.stsrcHospD.unique())
# print (df_2012.stsrcHospD.unique())
# print (df_2010.stsrcHospD.unique())
# # print (df_2018.columns.tolist())
# df_union = pd.concat([df_2010, df_2012,df_2014,df_2016,df_2018], ignore_index=True)
# print (df_union)
# print (df_union['surgyear'].value_counts())
# for col in df_union.columns:
#     print("Column '{}' have :: {}  missing values.".format(col,df_union[col].isna().sum()))
# df_union= pd.read_csv("df_union.csv")
# cols_to_remove = []
# samples = len(df_union)
# for col in df_union.columns:
#     nan_vals = df_union[col].isna().sum()
#     prec_missing_vals = nan_vals / samples
#     print("Column '{}' have :: {}  missing values.  {}%".format(col, df_union[col].isna().sum(), round(prec_missing_vals,3)))


# print (cols_to_remove)
#
# df_union.drop(cols_to_remove, axis=1, inplace=True)
# print("Number of Features : ",len(df_union.columns))
# for col in df_union.columns:
#     print("Column '{}' have :: {}  missing values.".format(col,df_union[col].isna().sum()))
#
# df_union.to_csv("df union after remove.csv")
# df_2018_ = pd.read_csv("/mnt/nadavrap-students/STS/data/2018_2019.csv")


df_all= pd.read_csv("/tmp/pycharm_project_723/df_union.csv")
print (df_all.reoperation.unique())
print (df_all.stsrcHospD.unique())
print (df_all.stsrcom.unique())
# mask = df_2018_['surgyear'] == 2018
# df_all = df_2018_[mask]
# mask_reop = df_all['reoperation'] == 1
# df_reop = df_all[mask_reop]
# df_op = df_all[~mask_reop]

def create_df_for_bins_hospid(col_mort):
    df1 = df_all.groupby(['hospid', 'surgyear'])['hospid'].count().reset_index(name='total')
    df2 = df_all.groupby(['hospid', 'surgyear'])['reoperation'].apply(lambda x: (x == 1).sum()).reset_index(
        name='Reop')
    df3 = df_all.groupby(['hospid', 'surgyear'])['reoperation'].apply(lambda x: (x == 0).sum()).reset_index(
        name='FirstOperation')
    df_aggr = pd.read_csv("aggregate_csv.csv")

    mask_reop = df_all['reoperation'] == 1
    df_reop = df_all[mask_reop]
    df_op = df_all[~mask_reop]

    dfmort = df_all.groupby(['hospid', 'surgyear'])[col_mort].apply(lambda x: (x == 1).sum()).reset_index(
        name='Mortality_all')
    dfmortf = df_op.groupby(['hospid', 'surgyear'])[col_mort].apply(lambda x: (x == 1).sum()).reset_index(
        name='Mortality_first')
    dfmortr = df_reop.groupby(['hospid', 'surgyear'])[col_mort].apply(lambda x: (x == 1).sum()).reset_index(
        name='Mortality_reop')

    df_comp = df_all.groupby(['hospid', 'surgyear'])['complics'].apply(lambda x: (x == 1).sum()).reset_index(
        name='Complics_all')
    df_compr = df_reop.groupby(['hospid', 'surgyear'])['complics'].apply(lambda x: (x == 1).sum()).reset_index(
        name='Complics_reop')
    df_compf = df_op.groupby(['hospid', 'surgyear'])['complics'].apply(lambda x: (x == 1).sum()).reset_index(
        name='Complics_FirstOperation')

    d1 = pd.merge(df1, df3, left_on=['hospid', 'surgyear'], right_on=['hospid', 'surgyear'], how='outer')
    d2 = pd.merge(d1, df2, left_on=['hospid', 'surgyear'], right_on=['hospid', 'surgyear'], how='outer')
    df5 = pd.merge(df_aggr, d2, left_on=['hospid', 'surgyear'], right_on=['hospid', 'surgyear'],
                   how='inner')  # how='left', on=['HospID','surgyear'])
    del df5["Unnamed: 0"]
    d3 = pd.merge(df5, dfmort, left_on=['hospid', 'surgyear'], right_on=['hospid', 'surgyear'], how='outer')
    d4 = pd.merge(d3, dfmortf, left_on=['hospid', 'surgyear'], right_on=['hospid', 'surgyear'], how='outer')
    d5 = pd.merge(d4, dfmortr, left_on=['hospid', 'surgyear'], right_on=['hospid', 'surgyear'], how='outer')
    d6 = pd.merge(d5, df_comp, left_on=['hospid', 'surgyear'], right_on=['hospid', 'surgyear'], how='outer')
    d7 = pd.merge(d6, df_compf, left_on=['hospid', 'surgyear'], right_on=['hospid', 'surgyear'], how='outer')
    d8 = pd.merge(d7, df_compr, left_on=['hospid', 'surgyear'], right_on=['hospid', 'surgyear'], how='outer')
    # df_sum_all_Years_total = pd.merge(d8, df_19, on='HospID', how='outer')
    d8.fillna(0, inplace=True)
    d8['mort_rate_All'] = (d8['Mortality_all'] / d8['total']) * 100
    d8['Mortality_First_rate'] = (d8['Mortality_first'] / d8['FirstOperation']) * 100
    d8['Mortality_Reop_rate'] = (d8['Mortality_reop'] / d8['Reop']) * 100
    d8['Complics_rate_All'] = (d8['Complics_all'] / d8['total']) * 100
    d8['Complics_First_rate'] = (d8['Complics_FirstOperation'] / d8['FirstOperation']) * 100
    d8['Complics_Reop_rate'] = (d8['Complics_reop'] / d8['Reop']) * 100
    d8.to_csv('hospid_year_allyears.csv')

    df_PredMort_all = df_all.groupby(['hospid', 'surgyear'])['predmort'].mean().reset_index(name='PredMort_All_avg')
    df_PredMort_op = df_op.groupby(['hospid', 'surgyear'])['predmort'].mean().reset_index(name='PredMort_First_avg')
    df_PredMort_reop = df_reop.groupby(['hospid', 'surgyear'])['predmort'].mean().reset_index(
        name='PredMort_Reoperation_avg')

    df_PredComp_all = df_all.groupby(['hospid', 'surgyear'])['predmm'].mean().reset_index(name='PredComp_All_avg')
    df_PredComp_op = df_op.groupby(['hospid', 'surgyear'])['predmm'].mean().reset_index(name='PredComp_First_avg')
    df_PredComp_reop = df_reop.groupby(['hospid', 'surgyear'])['predmm'].mean().reset_index(
        name='PredComp_Reoperation_avg')

    d19 = pd.merge(d8, df_PredMort_all, left_on=['hospid', 'surgyear'], right_on=['hospid', 'surgyear'], how='outer')
    d9 = pd.merge(d19, df_PredMort_op, left_on=['hospid', 'surgyear'], right_on=['hospid', 'surgyear'], how='outer')
    d10 = pd.merge(d9, df_PredMort_reop, left_on=['hospid', 'surgyear'], right_on=['hospid', 'surgyear'], how='outer')
    d14 = pd.merge(d10, df_PredComp_all, left_on=['hospid', 'surgyear'], right_on=['hospid', 'surgyear'], how='outer')
    d11 = pd.merge(d14, df_PredComp_op, left_on=['hospid', 'surgyear'], right_on=['hospid', 'surgyear'], how='outer')
    d12 = pd.merge(d11, df_PredComp_reop, left_on=['hospid', 'surgyear'], right_on=['hospid', 'surgyear'], how='outer')
    d12.fillna(0, inplace=True)

    d12['Mort_observe/expected_All'] = (d12['mort_rate_All'] / d12['PredMort_All_avg'])
    d12['Mort_observe/expected_First'] = (d12['Mortality_First_rate'] / d12['PredMort_First_avg'])
    d12['Mort_observe/expected_Reop'] = (d12['Mortality_Reop_rate'] / d12['PredMort_Reoperation_avg'])
    d12[['log_All_Mort', 'log_First_Mort', 'log_Reoperation_Mort']] = np.log2(
        d12[['Mort_observe/expected_All', 'Mort_observe/expected_First', 'Mort_observe/expected_Reop']].replace(0,
                                                                                                                np.nan))
    d12.fillna(0, inplace=True)

    d12['Comp_observe/expected_All'] = (d12['Complics_rate_All'] / d12['PredComp_All_avg'])
    d12['Comp_observe/expected_First'] = (d12['Complics_First_rate'] / d12['PredComp_First_avg'])
    d12['Comp_observe/expected_Reop'] = (d12['Complics_Reop_rate'] / d12['PredComp_Reoperation_avg'])
    d12[['log_All_Comp', 'log_First_Comp', 'log_Reoperation_Comp']] = np.log2(
        d12[['Comp_observe/expected_All', 'Comp_observe/expected_First', 'Comp_observe/expected_Reop']].replace(0,
                                                                                                                np.nan))
    d12.fillna(0, inplace=True)
    d12.to_csv("hospid_allyears_expec_hospid_stsrcHospD.csv")

    print(d12.info())
    print(d12.columns.tolist())


#create_df_for_bins_hospid('stsrcHospD')

def create_df_for_bins_surgid(col_mort):
    df1 = df_all.groupby(['surgid', 'surgyear'])['surgid'].count().reset_index(name='total')
    df2 = df_all.groupby(['surgid', 'surgyear'])['reoperation'].apply(lambda x: (x == 1).sum()).reset_index(
        name='Reop')
    df3 = df_all.groupby(['surgid', 'surgyear'])['reoperation'].apply(lambda x: (x == 0).sum()).reset_index(
        name='FirstOperation')
    df_aggr = pd.read_csv("/tmp/pycharm_project_723/aggregate_surgid_csv.csv")

    mask_reop = df_all['reoperation'] == 1
    df_reop = df_all[mask_reop]
    df_op = df_all[~mask_reop]

    dfmort = df_all.groupby(['surgid', 'surgyear'])[col_mort].apply(lambda x: (x == 1).sum()).reset_index(
        name='Mortality_all')
    dfmortf = df_op.groupby(['surgid', 'surgyear'])[col_mort].apply(lambda x: (x == 1).sum()).reset_index(
        name='Mortality_first')
    dfmortr = df_reop.groupby(['surgid', 'surgyear'])[col_mort].apply(lambda x: (x == 1).sum()).reset_index(
        name='Mortality_reop')

    df_comp = df_all.groupby(['surgid', 'surgyear'])['complics'].apply(lambda x: (x == 1).sum()).reset_index(
        name='Complics_all')
    df_compr = df_reop.groupby(['surgid', 'surgyear'])['complics'].apply(lambda x: (x == 1).sum()).reset_index(
        name='Complics_reop')
    df_compf = df_op.groupby(['surgid', 'surgyear'])['complics'].apply(lambda x: (x == 1).sum()).reset_index(
        name='Complics_FirstOperation')

    d1 = pd.merge(df1, df3, left_on=['surgid', 'surgyear'], right_on=['surgid', 'surgyear'], how='outer')
    d2 = pd.merge(d1, df2, left_on=['surgid', 'surgyear'], right_on=['surgid', 'surgyear'], how='outer')
    df5 = pd.merge(df_aggr, d2, left_on=['surgid', 'surgyear'], right_on=['surgid', 'surgyear'],how='inner')
    # del df5["Unnamed: 0"]
    d3 = pd.merge(df5, dfmort, left_on=['surgid', 'surgyear'], right_on=['surgid', 'surgyear'], how='outer')
    d4 = pd.merge(d3, dfmortf, left_on=['surgid', 'surgyear'], right_on=['surgid', 'surgyear'], how='outer')
    d5 = pd.merge(d4, dfmortr, left_on=['surgid', 'surgyear'], right_on=['surgid', 'surgyear'], how='outer')
    d6 = pd.merge(d5, df_comp, left_on=['surgid', 'surgyear'], right_on=['surgid', 'surgyear'], how='outer')
    d7 = pd.merge(d6, df_compf, left_on=['surgid', 'surgyear'], right_on=['surgid', 'surgyear'], how='outer')
    d8 = pd.merge(d7, df_compr, left_on=['surgid', 'surgyear'], right_on=['surgid', 'surgyear'], how='outer')
    # df_sum_all_Years_total = pd.merge(d8, df_19, on='HospID', how='outer')
    d8.fillna(0, inplace=True)
    d8['mort_rate_All'] = (d8['Mortality_all'] / d8['total']) * 100
    d8['Mortality_First_rate'] = (d8['Mortality_first'] / d8['FirstOperation']) * 100
    d8['Mortality_Reop_rate'] = (d8['Mortality_reop'] / d8['Reop']) * 100
    d8['Complics_rate_All'] = (d8['Complics_all'] / d8['total']) * 100
    d8['Complics_First_rate'] = (d8['Complics_FirstOperation'] / d8['FirstOperation']) * 100
    d8['Complics_Reop_rate'] = (d8['Complics_reop'] / d8['Reop']) * 100
    d8.to_csv('surgid_year_allyears.csv')

    df_PredMort_all = df_all.groupby(['surgid', 'surgyear'])['predmort'].mean().reset_index(name='PredMort_All_avg')
    df_PredMort_op = df_op.groupby(['surgid', 'surgyear'])['predmort'].mean().reset_index(name='PredMort_First_avg')
    df_PredMort_reop = df_reop.groupby(['surgid', 'surgyear'])['predmort'].mean().reset_index(
        name='PredMort_Reoperation_avg')

    df_PredComp_all = df_all.groupby(['surgid', 'surgyear'])['predmm'].mean().reset_index(name='PredComp_All_avg')
    df_PredComp_op = df_op.groupby(['surgid', 'surgyear'])['predmm'].mean().reset_index(name='PredComp_First_avg')
    df_PredComp_reop = df_reop.groupby(['surgid', 'surgyear'])['predmm'].mean().reset_index(
        name='PredComp_Reoperation_avg')

    d19 = pd.merge(d8, df_PredMort_all, left_on=['surgid', 'surgyear'], right_on=['surgid', 'surgyear'], how='outer')
    d9 = pd.merge(d19, df_PredMort_op, left_on=['surgid', 'surgyear'], right_on=['surgid', 'surgyear'], how='outer')
    d10 = pd.merge(d9, df_PredMort_reop, left_on=['surgid', 'surgyear'], right_on=['surgid', 'surgyear'], how='outer')
    d14 = pd.merge(d10, df_PredComp_all, left_on=['surgid', 'surgyear'], right_on=['surgid', 'surgyear'], how='outer')
    d11 = pd.merge(d14, df_PredComp_op, left_on=['surgid', 'surgyear'], right_on=['surgid', 'surgyear'], how='outer')
    d12 = pd.merge(d11, df_PredComp_reop, left_on=['surgid', 'surgyear'], right_on=['surgid', 'surgyear'], how='outer')
    d12.fillna(0, inplace=True)

    d12['Mort_observe/expected_All'] = (d12['mort_rate_All'] / d12['PredMort_All_avg'])
    d12['Mort_observe/expected_First'] = (d12['Mortality_First_rate'] / d12['PredMort_First_avg'])
    d12['Mort_observe/expected_Reop'] = (d12['Mortality_Reop_rate'] / d12['PredMort_Reoperation_avg'])
    d12[['log_All_Mort', 'log_First_Mort', 'log_Reoperation_Mort']] = np.log2(
        d12[['Mort_observe/expected_All', 'Mort_observe/expected_First', 'Mort_observe/expected_Reop']].replace(0,
                                                                                                                np.nan))
    d12.fillna(0, inplace=True)

    d12['Comp_observe/expected_All'] = (d12['Complics_rate_All'] / d12['PredComp_All_avg'])
    d12['Comp_observe/expected_First'] = (d12['Complics_First_rate'] / d12['PredComp_First_avg'])
    d12['Comp_observe/expected_Reop'] = (d12['Complics_Reop_rate'] / d12['PredComp_Reoperation_avg'])
    d12[['log_All_Comp', 'log_First_Comp', 'log_Reoperation_Comp']] = np.log2(
        d12[['Comp_observe/expected_All', 'Comp_observe/expected_First', 'Comp_observe/expected_Reop']].replace(0,
                                                                                                                np.nan))
    d12.fillna(0, inplace=True)
    d12.to_csv("surgid_allyears_expec_surgid_stsrcom.csv")

    print(d12.info())
    print(d12.columns.tolist())

# create_df_for_bins_surgid('stsrcom')

def add_Summary_Data_To_ImputedData(df):
    df1 = df.groupby(['hospid', 'surgyear'])['hospid'].count().reset_index(name='HospID_total_CABG')
    df2 = df.groupby(['hospid', 'surgyear'])['reoperation'].apply(lambda x: (x == 1).sum()).reset_index(name='HospID_Reop_CABG')

    df_aggr = pd.read_csv("aggregate_csv.csv")
    df3 = pd.merge(df1, df, left_on=['hospid','surgyear'], right_on=['hospid','surgyear'], how='outer')
    df4 = pd.merge(df2, df3, left_on=['hospid','surgyear'], right_on=['hospid','surgyear'], how='outer')
    df5 = pd.merge(df_aggr,df4,left_on=['hospid','surgyear'], right_on=['hospid','surgyear'], how='inner') #how='left', on=['HospID','surgyear'])

    # print(df5.info())
    # print(df5.isna().sum())
    # print(df5.columns.tolist())

    df_1 = df.groupby(['surgid', 'surgyear'])['surgid'].count().reset_index(name='surgid_total_CABG')
    df_2 = df.groupby(['surgid', 'surgyear'])['reoperation'].apply(lambda x: (x == 1).sum()).reset_index(
        name='surgid_Reop_CABG')

    df_aggrsurg = pd.read_csv("/tmp/pycharm_project_723/aggregate_surgid_csv.csv")
    df_aggrsurg.rename(columns={"total cardiac surgery": "surgid_total_cardiac_surgery"}, inplace=True)

    df6 = pd.merge(df_1, df_2, left_on=['surgid', 'surgyear'], right_on=['surgid', 'surgyear'], how='outer')
    df7 = pd.merge(df6,df_aggrsurg,left_on=['surgid','surgyear'], right_on=['surgid','surgyear'], how='inner')
    print ("=======================================")
    del df7["Unnamed: 0"]
    # print(df7.info())
    # print(df7.columns.tolist())
    df8 = pd.merge(df7,df5,left_on=['surgid','surgyear'], right_on=['surgid','surgyear'], how='inner')
    # print ("=======================================")
    # print(df8.head(10))
    # print(df8.shape)
    # print(df8.info())
    # print(df8.columns.tolist())
    del df8["Unnamed: 0_x"]
    del df8["Unnamed: 0_y"]
    df8.rename(columns={"total surgery count": "HospID_total_cardiac_surgery"}, inplace=True)
    df8[:1000].to_csv("/tmp/pycharm_project_723/new data sum info surg and Hosp 1000.csv")
    for col in df8.columns:
        print("Column '{}' have :: {}  missing values.".format(col, df8[col].isna().sum()))

    df8.to_csv("/tmp/pycharm_project_723/new data sum info surg and Hosp.csv")
    # df8.to_csv("/mnt/nadavrap-students/STS/data/imputed data sum info surg and Hosp.csv")
    return df8

#
# df_with_sum = add_Summary_Data_To_ImputedData(df_all)
# print(df_with_sum.shape)
# df_all= pd.read_csv("/tmp/pycharm_project_723/df_union.csv")
# hospidl =  []
# for col in df_all.hospid.unique():
#     print("col")