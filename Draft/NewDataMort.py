
from collections import Counter
import pandas as pd
import numpy as np
import os

import warnings
warnings.filterwarnings("ignore")


# df_all2 = pd.read_csv("/mnt/nadavrap-students/STS/data/imputed_data2.csv")
# df_all = pd.read_csv("/tmp/pycharm_project_723/imputed_data_with_numerical_values.csv")
#df_all = pd.read_csv("/tmp/pycharm_project_723/imputed_data_with_float_values_glmm.csv")


# df_all = df_all.replace({'Complics':{False:0, True:1}})
# df_all = df_all.replace({'Mortalty':{False:0, True:1}})
# df_all = df_all.replace({'PrCVInt':{False:0, True:1}})
# df_all = df_all.replace({'Mortality':{False:0, True:1}})
# df_all = df_all.replace({'Mt30Stat':{'Alive':0, 'Dead':1, np.nan:2}})
#
# df_all = df_all.replace({'Reoperation':{'First Time':0, 'Reoperation':1}})
# df_all.rename(columns={"EF<=35%": "EF_less_equal_35"}, inplace=True)
# df_all['HospID_total_cardiac_surgery'] = df_all['HospID_total_cardiac_surgery'].str.replace(',', '').astype(float)
#
#
#
# # df_test= df_all[
# #         ['HospID','HospID_total_cardiac_surgery', 'HospID_Reop_CABG', 'HospID_total_CABG', 'surgyear',
# #          'surgid', 'surgid_total_cardiac_surgery','surgid_total_CABG', 'surgid_Reop_CABG', 'SiteID',  'Complics', 'Mortalty']].copy()
# #
# # mask = df_all['surgyear'] == 2019 and df_all['surgyear'] == 2018
# # df_2019 = df_all[mask]
# # df_2019.to_csv("2018 2019.csv")
# # print (df_all.columns.tolist())
# # print (df_all.head(10))
# # df_all[:50].to_csv("numeric_df_after changes.csv")
# df_model_draft = df_all[
#         ['HospID','HospID_total_cardiac_surgery', 'HospID_Reop_CABG', 'HospID_total_CABG', 'surgyear',
#          'surgid', 'surgid_total_cardiac_surgery','surgid_total_CABG', 'surgid_Reop_CABG', 'SiteID',  'Age',
#          'Gender', 'RaceCaucasian', 'RaceBlack', 'RaceOther', 'Ethnicity',
#          'FHCAD', 'Diabetes', 'Hypertn', 'Dyslip', 'Dialysis', 'InfEndo', 'ChrLungD', 'ImmSupp', 'PVD', 'CreatLst',
#          'PrevMI', 'Arrhythmia', 'PrCVInt', 'POCPCI', 'MedACEI', 'MedASA',
#          'MedBeta', 'MedInotr', 'MedNitIV', 'MedSter', 'NumDisV', 'HDEF', 'VDInsufA', 'VDStenA', 'VDInsufM', 'VDStenM',
#          'VDInsufT', 'VDStenT', 'Status', 'SmokingStatus', 'InsulinDiab',
#          'ModSevereLungDis', 'PreCVAorTIAorCVD', 'RenFail', 'Angina', 'UnstableAngina', 'ClassNYHGroup',
#          'ArrhythAtrFibFlutter', 'ArrhythOther', 'DualAntiPlat', 'MedHeparin', 'AntiCoag',
#          'MedAntiplateltNoASA', 'NumDisV_ordinal', 'LeftMain', 'EF_less_equal_35', 'BMI',
#          'Complics', 'Mortality', 'Reoperation']].copy()
#
# # print (df_all['Mt30Stat'].unique())
# # counter = Counter(df_all['Mt30Stat'])
# # print(counter)
# # df1 = df_all.groupby(['surgyear'])['Mt30Stat'].apply(lambda x: (x == 1).sum()).reset_index(name='Dead')
# # df2 = df_all.groupby(['surgyear'])['surgyear'].count().reset_index(name='total')
# # df3 = df_all.groupby(['surgyear'])['Mt30Stat'].apply(lambda x: (x == 0).sum()).reset_index(name='Alive')
# # df6 = df_all.groupby(['surgyear'])['Mt30Stat'].apply(lambda x: (x == 2).sum()).reset_index(name='Nan')
# #
# # df4 = pd.merge(df2, df1, on='surgyear', how='outer')
# # df5 = pd.merge(df4, df3, on='surgyear', how='outer')
# # df7 = pd.merge(df5, df6, on='surgyear', how='outer')
# # print (df7.head(10))
# # df7.to_csv("summary of years surg Mt30Stat.csv")
# #
# # print(df_model_draft['Mortality'].value_counts())
# # print(df_model_draft['Complics'].value_counts())
#
#
# # df_train = df_model_draft[df_model_draft['surgyear'].isin([2010,2011,2012])]
# df_train = df_model_draft[df_model_draft['surgyear'].isin([2010,2011,2012,2013,2014,2015,2016])]
# df_test = df_model_draft[df_model_draft['surgyear'].isin([2017])]
# print(df_train['surgyear'].value_counts())
# print(df_test['surgyear'].value_counts())
#


##hilla
def check_mort_fields():
    df = pd.read_csv("/mnt/nadavrap-students/STS/data/2010_2011.csv")
    #df = df.replace({'stsrcHospD':{False:0, True:1}})
    #df = df.replace({'stsrcom':{False:0, True:1}})
    df = df.replace({'stsrcHospD':{'Alive':0, 'Dead':1}})
    #df = df.replace({'mtopd':{False:0, True:1}})
   # df = df.replace({'Mortalty':{False:0, True:1}})
    #df = df.replace({'Reoperation':{'First Time':0, 'Reoperation':1}})
    mask = df['surgyear'] == 2010
    df_new = df[mask]
    counter = Counter(df_new['stsrcHospD'])
    print(counter)
    df1 = df_new.groupby(['surgyear'])['stsrcHospD'].apply(lambda x: (x == 1).sum()).reset_index(name='Dead')
    df2 = df_new.groupby(['surgyear'])['surgyear'].count().reset_index(name='total')
    df3 = df_new.groupby(['surgyear'])['stsrcHospD'].apply(lambda x: (x == 0).sum()).reset_index(name='Alive')
    df6 = df_new.groupby(['surgyear'])['stsrcHospD'].apply(lambda x: (x == 2).sum()).reset_index(name='Nan')

    df4 = pd.merge(df2, df1, on='surgyear', how='outer')
    df5 = pd.merge(df4, df3, on='surgyear', how='outer')
    df7 = pd.merge(df5, df6, on='surgyear', how='outer')
    print (df7.head(10))
    df7.to_csv("stsrcHospD_2010.csv")

#check_mort_fields()

def sum_values(df_read,year,value):
    df_two_years = pd.read_csv(df_read)
    mask_first = df_two_years['surgyear'] == year
    df_new_f = df_two_years[mask_first]
    # mask_second = df_two_years['surgyear'] == 2011
    # df_new_sec = df_two_years[mask_second]

    df1 = df_new_f.groupby(['surgyear'])[value].apply(lambda x: (x == 1).sum()).reset_index(name='Dead')
    df2 = df_new_f.groupby(['surgyear'])[value].apply(lambda x: (x == 0).sum()).reset_index(name='Alive')
    # df3 = df_new_sec.groupby(['surgyear'])['stsrcHospD'].apply(lambda x: (x == 1).sum()).reset_index(name='Alive')
    # df4 = df_new_sec.groupby(['surgyear'])['stsrcHospD'].apply(lambda x: (x == 0).sum()).reset_index(name='Dead')
    df5 = pd.merge(df1, df2, on='surgyear', how='outer')
    # df6 = pd.merge(df3, df4, on='surgyear', how='outer')
    print(year,df5)
    # print(df6)
#####stsrcom
def running(value):
    sum_values("/mnt/nadavrap-students/STS/data/2010_2011.csv",2010,value)
    sum_values("/mnt/nadavrap-students/STS/data/2010_2011.csv",2011,value)
    sum_values("/mnt/nadavrap-students/STS/data/2012_2013.csv",2012,value)
    sum_values("/mnt/nadavrap-students/STS/data/2012_2013.csv",2013,value)
    sum_values("/mnt/nadavrap-students/STS/data/2014_2015.csv",2014,value)
    sum_values("/mnt/nadavrap-students/STS/data/2014_2015.csv",2015,value)
    sum_values("/mnt/nadavrap-students/STS/data/2016_2017.csv",2016,value)
    sum_values("/mnt/nadavrap-students/STS/data/2016_2017.csv",2017,value)
    sum_values("/mnt/nadavrap-students/STS/data/2018_2019.csv",2018,value)
    sum_values("/mnt/nadavrap-students/STS/data/2018_2019.csv",2019,value)
# running("stsrcHospD")
running("stsrcom")
# print(df_new_11)


