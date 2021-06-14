import pandas as pd
from sklearn import linear_model
import statsmodels.api as sm
import numpy as np
from scipy import stats


df_all = pd.read_csv("/mnt/nadavrap-students/STS/data/imputed_data2.csv")

# df_all = pd.read_csv("/tmp/pycharm_project_723/new data sum info surg and Hosp numeric values.csv")



# # print(df_all.columns.tolist())
# # print (df_all.count())
# # print (df_all['Mortalty'].isnull().sum())
# # print (df_all['Mortalty'].value_counts())
# def refactor_categorical_values_to_numeric_values(df, col_names):
#     # df = df.filter(col_names, axis=1)
#     for col in col_names:
#         try:
#             df = df.replace({col: {False: 0, True: 1}})
#             df = df.replace({col: {"No": 0, "Yes": 1}})
#             df = df.replace({col: {"Male": 0, "Female": 1}})
#             df = df.replace({col: {"Elective": 0, "Urgent": 1}})
#             df = df.replace({col: {"Non-Hispanic": 0, "Hispanic": 1}})
#             df = df.replace({col: {"Previous Non-CAB": 0, "Previous CAB": 1}})
#             df = df.replace({col: {"None/Trivial/Trace/Mild": 0, "Moderate/Severe": 1}})
#             df = df.replace({col: {"Unknown": 1, "Alive": 1, "Dead": 0}})
#             df = df.replace({col: {"First cardiovascular surgery": 0, "NA - Not a cardiovascular surgery": 0,
#                                    "First re-op cardiovascular surgery": 0, "Second re-op cardiovascular surgery": 1,
#                                    "Third re-op cardiovascular surgery": 1,
#                                    "Fourth or more re-op cardiovascular surgery": 1}})
#             df = df.replace({col: {"Never smoker": 0, "Smoker": 1}})
#             df = df.replace({col: {"I/II": 0, "III/IV": 1}})
#             df = df.replace({col: {"None": 0, "One": 1, "Two": 2, "Three": 3}})
#         except:
#             x = "none"
#     print(df.shape)
#     df.to_csv("/tmp/pycharm_project_723/new data sum info surg and Hosp numeric values.csv")
#
df_all = df_all.replace({'STSRCHOSPD':{False:0, True:1}})
df_all = df_all.replace({'Complics':{False:0, True:1}})
df_all = df_all.replace({'Mortality':{False:0, True:1}})
df_all = df_all.replace({'STSRCMM':{False:0, True:1}})


print (df_all['STSRCMM'].unique())
print (df_all['STSRCMM'].isna().sum())
df_all[:50].to_csv("all 50.csv")# def intersection(lst1, lst2):
#     return list(set(lst1) & set(lst2))
#
#
# # list_vals = [ "Reoperation", "BMI", "Age", "Gender", "RaceCaucasian", "RaceBlack", "Ethnicity",
# #                  "RaceOther", "FHCAD", "Diabetes", "InsulinDiab", "Dyslip", "Dialysis", "Hypertn", "InfEndo",
# #                  "SmokingStatus", "ChrLungD", "ModSevereLungDis", "ImmSupp", "PVD", "DualAntiPlat", 'RenFail',
# #                  "CreatLst", 'PreCVAorTIAorCVD', "POCPCI", "PrevMI", "Angina", "UnstableAngina", "HeartFail",
# #                  "ClassNYHGroup", "Arrhythmia", "ArrhythAtrFibFlutter", "ArrhythOther", "MedACEI", "MedBeta",
# #                  "MedNitIV", "MedASA", "MedAntiplateltNoASA", "AntiCoag", "MedInotr", "MedSter", "HDEF", "EF<=35%",
# #                  "NumDisV", 'NumDisV_ordinal', "LeftMain", "VDInsufA", "VDStenA", "VDInsufM", "VDStenM", "VDInsufT",
# #                  "VDStenT", "Status", 'MedHeparin', 'Mortality', 'PrCVInt']
# #     # list_val = ['PrCVInt']
# #
# #
# # # print (intersection(list2,list_vals))
# # test = df_all[:50]
# # refactor_categorical_values_to_numeric_values(test, list_vals)
# # test.rename(columns={"EF<=35%": "EF_less_equal_35"}, inplace=True)
# list2 =  [ 'STSRCHOSPD', 'STSRCOM', 'STSRCDSWI', 'STSRCMM', 'STSRCPermStroke', 'STSRCProlvent', 'STSRcRenFail', 'STSRCreop',
#            'PLOS', 'PredMort', 'PredDeep', 'PredReop', 'PredStro', 'PredVent', 'PredRenF', 'PredMM', 'Pred6D', 'Pred14D'
#            'Age', 'Gender', 'RaceCaucasian', 'RaceBlack', 'RaceOther', 'Ethnicity', 'FHCAD', 'Diabetes', 'Hypertn',
#            'Dyslip', 'Dialysis', 'InfEndo', 'ChrLungD', 'ImmSupp', 'PVD', 'CreatLst', 'PrevMI', 'Arrhythmia', 'PrCVInt', 'prcab',
#            'prvalve', 'POCPCI', 'ProthCar', 'MedACEI', 'MedASA', 'MedBeta', 'MedInotr', 'MedNitIV', 'MedSter', 'NumDisV', 'HDEF',
#            'VDInsufA', 'VDStenA', 'VDInsufM', 'VDStenM', 'VDInsufT', 'VDStenT', 'Status', 'PerfusTm', 'XClampTm', 'DistVein', 'NumIMADA',
#            'NumRadDA', 'IABP', 'VentHrsTot', 'Complics', 'COpReBld', 'CPVntLng', 'CRenFail', 'HeartFail', 'Incidenc', 'Reoperation',
#            'SmokingStatus', 'InsulinDiab', 'ModSevereLungDis', 'PreCVAorTIAorCVD', 'RenFail', 'Angina', 'UnstableAngina', 'ClassNYHGroup',
#            'ArrhythAtrFibFlutter', 'ArrhythOther', 'DualAntiPlat', 'MedHeparin', 'AntiCoag', 'MedAntiplateltNoASA', 'NumDisV_ordinal', 'EF<=35%',
#            'CPBUse', 'RadArtUsed', 'IMAGraftUsed', 'DistVeinDone', 'TotalNumberOfGrafts', 'LeftMain', 'CompleteRevas', 'MajorComps', 'PLOS14',
#            'postCVAorTIA', 'IntraPostBloodTrans', 'ICUHrsTotal', 'BMI']
# # list2.to_csv("test for numeric draft model.csv")
# refactor_categorical_values_to_numeric_values(df_all,list2)


# mask_reop = df_all['Reoperation'] == 'Reoperation'
# df_reop = df_all[mask_reop]
# df_all = df_all.replace({'Reoperation':{'First Time':0, 'Reoperation':1}})
mask = df_all['surgyear'] == 2010
df_2010 = df_all[mask]
mask = df_all['surgyear'] == 2011
df_2011 = df_all[mask]
mask = df_all['surgyear'] == 2012
df_2012 = df_all[mask]
mask = df_all['surgyear'] == 2013
df_2013 = df_all[mask]
mask = df_all['surgyear'] == 2014
df_2014 = df_all[mask]
mask = df_all['surgyear'] == 2015
df_2015 = df_all[mask]
mask = df_all['surgyear'] == 2016
df_2016 = df_all[mask]
mask = df_all['surgyear'] == 2017
df_2017 = df_all[mask]
mask = df_all['surgyear'] == 2018
df_2018 = df_all[mask]
mask = df_all['surgyear'] == 2019
df_2019 = df_all[mask]

# hospid_2019 = pd.DataFrame()
# mask = df_all['HospID'] == 100427
# df1 = df_all[mask]
# df1.to_csv('100427.csv')
# df2 = df1.groupby(['HospID','surgyear'])['HospID'].count().reset_index(name='total')
# print (df2.head(6))


def create_2019_df(df):
    df1 = df.groupby('HospID')['HospID'].count().reset_index(name='total')
    df2 = df.groupby('HospID')['Reoperation'].apply(lambda x: (x == 'Reoperation').sum()).reset_index(name='Reop')
    df3 = df.groupby('HospID')['Reoperation'].apply(lambda x: (x == 'First Time').sum()).reset_index(name='FirstOperation')
    dfmort = df.groupby('HospID')['MtOpD'].apply(lambda x: (x == 1).sum()).reset_index(name='Mortality_all')

    mask_reop = df['Reoperation'] == 'Reoperation'
    df_reop = df[mask_reop]
    df_op = df[~mask_reop]
    dfmortf = df_op.groupby('HospID')['MtOpD'].apply(lambda x: (x == 1).sum()).reset_index(name='Mortality_first')
    dfmortr = df_reop.groupby('HospID')['MtOpD'].apply(lambda x: (x == 1).sum()).reset_index(name='Mortality_reop')

    df_comp = df.groupby('HospID')['Complics'].apply(lambda x: (x == 1).sum()).reset_index(name='Complics_all')
    df_compr = df_reop.groupby('HospID')['Complics'].apply(lambda x: (x == 1).sum()).reset_index(name='Complics_reop')
    df_compf = df_op.groupby('HospID')['Complics'].apply(lambda x: (x == 1).sum()).reset_index(name='Complics_FirstOperation')

    d1 = pd.merge(df1, df3, on='HospID', how='outer')
    d2 = pd.merge(d1, df2, on='HospID', how='outer')
    d3 = pd.merge(d2, dfmort, on='HospID', how='outer')
    d4 = pd.merge(d3, dfmortf, on='HospID', how='outer')
    d5 = pd.merge(d4, dfmortr, on='HospID', how='outer')
    d6 = pd.merge(d5, df_comp, on='HospID', how='outer')
    d7 = pd.merge(d6, df_compf, on='HospID', how='outer')
    d8 = pd.merge(d7, df_compr, on='HospID', how='outer')
    #df_sum_all_Years_total = pd.merge(d8, df_19, on='HospID', how='outer')
    d8.fillna(0, inplace=True)
    d8['mort_rate_All'] = (d8['Mortality_all'] / d8['total'])*100
    d8['Mortality_First_rate'] =( d8['Mortality_first'] / d8['FirstOperation'])*100
    d8['Mortality_Reop_rate'] = (d8['Mortality_reop'] / d8['Reop'])*100
    d8['Complics_rate_All'] = (d8['Complics_all'] / d8['total']) * 100
    d8['Complics_First_rate'] = (d8['Complics_FirstOperation'] / d8['FirstOperation']) * 100
    d8['Complics_Reop_rate'] = (d8['Complics_reop'] / d8['Reop']) * 100
    d8.to_csv("oneyear_hospid.csv")

    df_PredMort_op = df_op.groupby('HospID')['PredMort'].mean().reset_index(name='PredMort_First_avg')
    df_PredMort_reop= df_reop.groupby('HospID')['PredMort'].mean().reset_index(name='PredMort_Reoperation_avg')

    df_PredComp_op = df_op.groupby('HospID')['PredMM'].mean().reset_index(name='PredComp_First_avg')
    df_PredComp_reop= df_reop.groupby('HospID')['PredMM'].mean().reset_index(name='PredComp_Reoperation_avg')

    d9 = pd.merge(d8, df_PredMort_op, on='HospID', how='outer')
    d10 = pd.merge(d9, df_PredMort_reop, on='HospID', how='outer')
    d11 = pd.merge(d10, df_PredComp_op, on='HospID', how='outer')
    d12 = pd.merge(d11, df_PredComp_reop, on='HospID', how='outer')
    d12.fillna(0, inplace=True)
    d12['Mort_observe/expected_First'] = (d12['Mortality_First_rate'] / d12['PredMort_First_avg'])
    d12['Mort_observe/expected_Reop'] = (d12['Mortality_Reop_rate'] / d12['PredMort_Reoperation_avg'])
    d12[['log_First_Mort', 'log_Reoperation_Mort']] = np.log2(
    d12[['Mort_observe/expected_First', 'Mort_observe/expected_Reop']].replace(0, np.nan))
    d12.fillna(0, inplace=True)

    d12['Comp_observe/expected_First'] = (d12['Complics_First_rate'] / d12['PredComp_First_avg'])
    d12['Comp_observe/expected_Reop'] = (d12['Complics_Reop_rate'] / d12['PredComp_Reoperation_avg'])
    d12[['log_First_Comp', 'log_Reoperation_Comp']] = np.log2(
    d12[['Comp_observe/expected_First', 'Comp_observe/expected_Reop']].replace(0, np.nan))
    d12.to_csv("oneyear_expec_hospid.csv")



def create_df():
    df1 = df_all.groupby(['HospID','surgyear'])['HospID'].count().reset_index(name='total')
    df2 = df_all.groupby(['HospID','surgyear'])['Reoperation'].apply(lambda x: (x == 'Reoperation').sum()).reset_index(name='Reop')
    df3 = df_all.groupby(['HospID','surgyear'])['Reoperation'].apply(lambda x: (x == 'First Time').sum()).reset_index(name='FirstOperation')
    df_aggr = pd.read_csv("aggregate_csv.csv")

    mask_reop = df_all['Reoperation'] == 'Reoperation'
    df_reop = df_all[mask_reop]
    df_op = df_all[~mask_reop]

    dfmort = df_all.groupby(['HospID','surgyear'])['STSRCMM'].apply(lambda x: (x == 1).sum()).reset_index(name='Mortality_all')
    dfmortf = df_op.groupby(['HospID','surgyear'])['STSRCMM'].apply(lambda x: (x == 1).sum()).reset_index(name='Mortality_first')
    dfmortr = df_reop.groupby(['HospID','surgyear'])['STSRCMM'].apply(lambda x: (x == 1).sum()).reset_index(name='Mortality_reop')

    df_comp = df_all.groupby(['HospID','surgyear'])['Complics'].apply(lambda x: (x == 1).sum()).reset_index(name='Complics_all')
    df_compr = df_reop.groupby(['HospID','surgyear'])['Complics'].apply(lambda x: (x == 1).sum()).reset_index(name='Complics_reop')
    df_compf = df_op.groupby(['HospID','surgyear'])['Complics'].apply(lambda x: (x == 1).sum()).reset_index(name='Complics_FirstOperation')

    d1 = pd.merge(df1, df3,  left_on=['HospID','surgyear'], right_on=['HospID','surgyear'], how='outer')
    d2 = pd.merge(d1, df2,  left_on=['HospID','surgyear'], right_on=['HospID','surgyear'], how='outer')
    df5 = pd.merge(df_aggr, d2, left_on=['HospID', 'surgyear'], right_on=['HospID', 'surgyear'],
                   how='inner')  # how='left', on=['HospID','surgyear'])
    del df5["Unnamed: 0"]
    d3 = pd.merge(df5, dfmort, left_on=['HospID','surgyear'], right_on=['HospID','surgyear'], how='outer')
    d4 = pd.merge(d3, dfmortf,left_on=['HospID','surgyear'], right_on=['HospID','surgyear'],how='outer')
    d5 = pd.merge(d4, dfmortr,left_on=['HospID','surgyear'], right_on=['HospID','surgyear'],how='outer')
    d6 = pd.merge(d5, df_comp,left_on=['HospID','surgyear'], right_on=['HospID','surgyear'],how='outer')
    d7 = pd.merge(d6, df_compf, left_on=['HospID','surgyear'], right_on=['HospID','surgyear'], how='outer')
    d8 = pd.merge(d7, df_compr, left_on=['HospID','surgyear'], right_on=['HospID','surgyear'], how='outer')
    # df_sum_all_Years_total = pd.merge(d8, df_19, on='HospID', how='outer')
    d8.fillna(0, inplace=True)
    d8['mort_rate_All'] = (d8['Mortality_all'] / d8['total'])*100
    d8['Mortality_First_rate'] =( d8['Mortality_first'] / d8['FirstOperation'])*100
    d8['Mortality_Reop_rate'] = (d8['Mortality_reop'] / d8['Reop'])*100
    d8['Complics_rate_All'] = (d8['Complics_all'] / d8['total']) * 100
    d8['Complics_First_rate'] = (d8['Complics_FirstOperation'] / d8['FirstOperation']) * 100
    d8['Complics_Reop_rate'] = (d8['Complics_reop'] / d8['Reop']) * 100
    d8.to_csv('hospid_year_allyears.csv')

    df_PredMort_all = df_all.groupby(['HospID','surgyear'])['PredMort'].mean().reset_index(name='PredMort_All_avg')
    df_PredMort_op = df_op.groupby(['HospID','surgyear'])['PredMort'].mean().reset_index(name='PredMort_First_avg')
    df_PredMort_reop = df_reop.groupby(['HospID','surgyear'])['PredMort'].mean().reset_index(name='PredMort_Reoperation_avg')

    df_PredComp_all = df_all.groupby(['HospID','surgyear'])['PredMM'].mean().reset_index(name='PredComp_All_avg')
    df_PredComp_op = df_op.groupby(['HospID','surgyear'])['PredMM'].mean().reset_index(name='PredComp_First_avg')
    df_PredComp_reop = df_reop.groupby(['HospID','surgyear'])['PredMM'].mean().reset_index(name='PredComp_Reoperation_avg')

    d19 = pd.merge(d8, df_PredMort_all, left_on=['HospID','surgyear'], right_on=['HospID','surgyear'], how='outer')
    d9 = pd.merge(d19, df_PredMort_op, left_on=['HospID','surgyear'], right_on=['HospID','surgyear'], how='outer')
    d10 = pd.merge(d9, df_PredMort_reop, left_on=['HospID','surgyear'], right_on=['HospID','surgyear'], how='outer')
    d14 = pd.merge(d10, df_PredComp_all, left_on=['HospID','surgyear'], right_on=['HospID','surgyear'], how='outer')
    d11 = pd.merge(d14, df_PredComp_op, left_on=['HospID','surgyear'], right_on=['HospID','surgyear'], how='outer')
    d12 = pd.merge(d11, df_PredComp_reop, left_on=['HospID','surgyear'], right_on=['HospID','surgyear'], how='outer')
    d12.fillna(0, inplace=True)

    d12['Mort_observe/expected_All'] = (d12['mort_rate_All'] / d12['PredMort_All_avg'])
    d12['Mort_observe/expected_First'] = (d12['Mortality_First_rate'] / d12['PredMort_First_avg'])
    d12['Mort_observe/expected_Reop'] = (d12['Mortality_Reop_rate'] / d12['PredMort_Reoperation_avg'])
    d12[['log_All_Mort','log_First_Mort','log_Reoperation_Mort']] = np.log2(
        d12[['Mort_observe/expected_All','Mort_observe/expected_First', 'Mort_observe/expected_Reop']].replace(0, np.nan))
    d12.fillna(0, inplace=True)

    d12['Comp_observe/expected_All'] = (d12['Complics_rate_All'] / d12['PredComp_All_avg'])
    d12['Comp_observe/expected_First'] = (d12['Complics_First_rate'] / d12['PredComp_First_avg'])
    d12['Comp_observe/expected_Reop'] = (d12['Complics_Reop_rate'] / d12['PredComp_Reoperation_avg'])
    d12[['log_All_Comp','log_First_Comp', 'log_Reoperation_Comp']] = np.log2(
        d12[['Comp_observe/expected_All','Comp_observe/expected_First', 'Comp_observe/expected_Reop']].replace(0, np.nan))
    d12.fillna(0, inplace=True)
    d12.to_csv("hospid_allyears_expec_hospid_STSRCMM.csv")

    print(d12.info())
    print(d12.columns.tolist())



# create_df()
# create_2019_df(df_2017)
# def total_years(row,hospid,year):
#     print (row)
#     print (hospid)
#     print (year)
#     row_index = df_all_surg[df_all_surg['Hospital ID'] == hospid].index.item()
#     print (row_index)
#     return df_all_surg.loc[row_index,year]
# data = []
def create_aggr_df():
    df_all_surg = pd.read_csv("AggregateAllSurg.csv")
    # print(df_all_surg.columns.tolist())
    print (df_all_surg.head(10))
    for index, row in df_all_surg.iterrows():
        l0 = [row['Hospital ID'], 2010, row["2010"]]
        l1 = [row['Hospital ID'], 2011, row["2011"]]
        l2 = [row['Hospital ID'], 2012, row["2012"]]
        l3 = [row['Hospital ID'], 2013, row["2013"]]
        l4 = [row['Hospital ID'], 2014, row["2014"]]
        l5 = [row['Hospital ID'], 2015, row["2015"]]
        l6 = [row['Hospital ID'], 2016, row["2016"]]
        l7 = [row['Hospital ID'], 2017, row["2017"]]
        l8 = [row['Hospital ID'], 2018, row["2018"]]
        l9 = [row['Hospital ID'], 2019, row["2019"]]
        data.append(l0)
        data.append(l1)
        data.append(l2)
        data.append(l3)
        data.append(l4)
        data.append(l5)
        data.append(l6)
        data.append(l7)
        data.append(l8)
        data.append(l9)
    print (data)
    # data_transposed = zip(data)
    # df = pd.DataFrame(data_transposed, columns=["HospID", "surgyear", "count"])
    df = pd.DataFrame(data,columns=["hospid", "surgyear", "total cardiac surgery"])
    df.to_csv("aggregate_csv.csv")



def add_Summary_Data_To_ImputedData(df):
    df1 = df.groupby(['HospID', 'surgyear'])['HospID'].count().reset_index(name='HospID_total_CABG')
    df2 = df.groupby(['HospID', 'surgyear'])['Reoperation'].apply(lambda x: (x == 1).sum()).reset_index(name='HospID_Reop_CABG')

    df_aggr = pd.read_csv("aggregate_csv.csv")
    df3 = pd.merge(df1, df, left_on=['HospID','surgyear'], right_on=['HospID','surgyear'], how='outer')
    df4 = pd.merge(df2, df3, left_on=['HospID','surgyear'], right_on=['HospID','surgyear'], how='outer')
    df5 = pd.merge(df_aggr,df4,left_on=['HospID','surgyear'], right_on=['HospID','surgyear'], how='inner') #how='left', on=['HospID','surgyear'])
    del df5["Unnamed: 0"]
    # print(df5.info())
    # print(df5.columns.tolist())

    df_1 = df.groupby(['surgid', 'surgyear'])['surgid'].count().reset_index(name='surgid_total_CABG')
    df_2 = df.groupby(['surgid', 'surgyear'])['Reoperation'].apply(lambda x: (x == 1).sum()).reset_index(
        name='surgid_Reop_CABG')

    df_aggrsurg = pd.read_csv("/tmp/pycharm_project_723/aggregate_surgid_csv.csv")
    df_aggrsurg.rename(columns={"total cardiac surgery": "surgid_total_cardiac_surgery"}, inplace=True)

    df6 = pd.merge(df_1, df_2, left_on=['surgid', 'surgyear'], right_on=['surgid', 'surgyear'], how='outer')
    df7 = pd.merge(df6,df_aggrsurg,left_on=['surgid','surgyear'], right_on=['surgid','surgyear'], how='inner')

    del df7["Unnamed: 0"]
    # print(df7.info())
    # print(df7.columns.tolist())
    df8 = pd.merge(df5,df7,left_on=['surgid','surgyear'], right_on=['surgid','surgyear'], how='inner')
    # print ("=======================================")
    # print(df8.head(10))
    # print(df8.shape)
    # print(df8.info())
    # print(df8.columns.tolist())
    df8.rename(columns={"total surgery count": "HospID_total_cardiac_surgery"}, inplace=True)

    df8.to_csv("/tmp/pycharm_project_723/new data sum info surg and Hosp.csv")
    print (df8.columns.tolist())
    return df8



# df_with_sum = add_Summary_Data_To_ImputedData(df_all)
# print(df_with_sum.shape)
# create_aggr_df()

def create_aggr_surgid_df():
    df_all_surg = pd.read_csv("/tmp/pycharm_project_723/surgid_before_aggregate.csv")
    # print(df_all_surg.columns.tolist())
    print (df_all_surg.head(10))
    for index, row in df_all_surg.iterrows():
        l0 = [row['Surgeon ID'], 2010, row["2010"]]
        l1 = [row['Surgeon ID'], 2011, row["2011"]]
        l2 = [row['Surgeon ID'], 2012, row["2012"]]
        l3 = [row['Surgeon ID'], 2013, row["2013"]]
        l4 = [row['Surgeon ID'], 2014, row["2014"]]
        l5 = [row['Surgeon ID'], 2015, row["2015"]]
        l6 = [row['Surgeon ID'], 2016, row["2016"]]
        l7 = [row['Surgeon ID'], 2017, row["2017"]]
        l8 = [row['Surgeon ID'], 2018, row["2018"]]
        l9 = [row['Surgeon ID'], 2019, row["2019"]]
        data.append(l0)
        data.append(l1)
        data.append(l2)
        data.append(l3)
        data.append(l4)
        data.append(l5)
        data.append(l6)
        data.append(l7)
        data.append(l8)
        data.append(l9)
    print (data)
    # data_transposed = zip(data)
    # df = pd.DataFrame(data_transposed, columns=["HospID", "surgyear", "count"])
    df = pd.DataFrame(data,columns=["surgid", "surgyear", "total cardiac surgery"])
    df.to_csv("aggregate_surgid_csv.csv")

def add_Summary_surgid_Data_To_ImputedData(df):
    df1 = df.groupby(['surgid', 'surgyear'])['surgid'].count().reset_index(name='surgid_total_CABG')
    df2 = df.groupby(['surgid', 'surgyear'])['Reoperation'].apply(lambda x: (x == 'Reoperation').sum()).reset_index(name='surgid_Reop_CABG')
    df_aggr = pd.read_csv("/tmp/pycharm_project_723/aggregate_surgid_csv.csv")
    df_aggr.rename(columns={"total cardiac surgery": "surgid_total_cardiac_surgery"}, inplace=True)

    df3 = pd.merge(df1, df, left_on=['surgid','surgyear'], right_on=['surgid','surgyear'], how='outer')
    df4 = pd.merge(df2, df3, left_on=['surgid','surgyear'], right_on=['surgid','surgyear'], how='outer')
    df5 = pd.merge(df_aggr,df4,left_on=['surgid','surgyear'], right_on=['surgid','surgyear'], how='inner') #how='left', on=['HospID','surgyear'])


    print(df5.head(10))
    print(df5.shape)
    print(df5.info())
    print(df5.columns.tolist())
    df5.to_csv("/tmp/pycharm_project_723/imputed data with surgid sum info.csv")
    return df5

# df_sum = pd.read_csv("/tmp/pycharm_project_723/imputed_data_with_float_values_glmm.csv")
# add_Summary_surgid_Data_To_ImputedData(df_sum)


def create_surgid_df():
    df1 = df_all.groupby(['surgid', 'surgyear'])['surgid'].count().reset_index(name='total')
    df2 = df_all.groupby(['surgid', 'surgyear'])['Reoperation'].apply(lambda x: (x == 'Reoperation').sum()).reset_index(
        name='Reop')
    df3 = df_all.groupby(['surgid', 'surgyear'])['Reoperation'].apply(lambda x: (x == 'First Time').sum()).reset_index(
        name='FirstOperation')
    df_aggr = pd.read_csv("/tmp/pycharm_project_723/aggregate_surgid_csv.csv")

    mask_reop = df_all['Reoperation'] == 'Reoperation'
    df_reop = df_all[mask_reop]
    df_op = df_all[~mask_reop]

    dfmort = df_all.groupby(['surgid', 'surgyear'])['STSRCMM'].apply(lambda x: (x == 1).sum()).reset_index(
        name='Mortality_all')
    dfmortf = df_op.groupby(['surgid', 'surgyear'])['STSRCMM'].apply(lambda x: (x == 1).sum()).reset_index(
        name='Mortality_first')
    dfmortr = df_reop.groupby(['surgid', 'surgyear'])['STSRCMM'].apply(lambda x: (x == 1).sum()).reset_index(
        name='Mortality_reop')

    df_comp = df_all.groupby(['surgid', 'surgyear'])['Complics'].apply(lambda x: (x == 1).sum()).reset_index(
        name='Complics_all')
    df_compr = df_reop.groupby(['surgid', 'surgyear'])['Complics'].apply(lambda x: (x == 1).sum()).reset_index(
        name='Complics_reop')
    df_compf = df_op.groupby(['surgid', 'surgyear'])['Complics'].apply(lambda x: (x == 1).sum()).reset_index(
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

    df_PredMort_all = df_all.groupby(['surgid', 'surgyear'])['PredMort'].mean().reset_index(name='PredMort_All_avg')
    df_PredMort_op = df_op.groupby(['surgid', 'surgyear'])['PredMort'].mean().reset_index(name='PredMort_First_avg')
    df_PredMort_reop = df_reop.groupby(['surgid', 'surgyear'])['PredMort'].mean().reset_index(
        name='PredMort_Reoperation_avg')

    df_PredComp_all = df_all.groupby(['surgid', 'surgyear'])['PredMM'].mean().reset_index(name='PredComp_All_avg')
    df_PredComp_op = df_op.groupby(['surgid', 'surgyear'])['PredMM'].mean().reset_index(name='PredComp_First_avg')
    df_PredComp_reop = df_reop.groupby(['surgid', 'surgyear'])['PredMM'].mean().reset_index(
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
    d12.to_csv("surgid_allyears_expec_surgid_STSRCMM.csv")

    print(d12.info())
    print(d12.columns.tolist())

create_df()
create_surgid_df()
# df = pd.read_csv("/tmp/pycharm_project_723/test_all.csv")
# df.to_csv("/mnt/nadavrap-students/STS/data/imputed data with sum info.csv")

