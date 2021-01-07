import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

df=pd.read_csv("/mnt/nadavrap-students/STS/data/imputed_data2.csv")

df = df.drop(columns=['SiteID', 'surgyear','surgid'], axis=1)

list_vals=['Reoperation','Age','CreatLst','BMI','ICUHrsTotal','Gender','NumDisV', 'Ethnicity', 'FHCAD', 'RaceCaucasian', 'RaceBlack', 'RaceOther',
           'Hypertn', 'Dyslip', 'Diabetes', 'InsulinDiab', 'PVD', 'PrCVInt', 'SmokingStatus', 'ChrLungD',
           'ModSevereLungDis', 'RenFail', 'Dialysis', 'ImmSupp', 'InfEndo', 'PrevMI',
           'POCPCI', 'Angina', 'UnstableAngina', 'HeartFail', 'ClassNYHGroup', 'Arrhythmia', 'ArrhythAtrFibFlutter',
           'ArrhythOther', 'HDEF', 'NumDisV_ordinal', 'LeftMain', 'VDStenA', 'VDInsufA',
           'VDStenM', 'VDInsufM', 'VDStenT', 'AntiCoag', 'MedASA', 'MedAntiplateltNoASA', 'DualAntiPlat', 'MedHeparin', 'MedNitIV', 'MedInotr',
           'MedACEI', 'MedBeta', 'MedSter']

#
df = df.filter(list_vals, axis=1)
df = df.replace({'Reoperation': {'First Time': 0, 'Reoperation': 1}})
df = df.replace({'Gender': {"Male": 0, "Female": 1}})
for col in list_vals:
    try:
        df = df.replace({col: {False: 0, True: 1}})
        df = df.replace({col: {"No": 0, "Yes": 1}})
        df = df.replace({col: {"Male": 0, "Female": 1}})
        df = df.replace({col: {"Elective": 0, "Urgent": 1}})
        df = df.replace({col: {"Non-Hispanic": 0, "Hispanic": 1}})
        df = df.replace({col: {"Previous Non-CAB": 0, "Previous CAB": 1}})
        df = df.replace({col: {"None/Trivial/Trace/Mild": 0, "Moderate/Severe": 1}})
        df = df.replace({col: {"Unknown": 1, "Alive": 1, "Dead": 0}})
        df = df.replace({col: {"First cardiovascular surgery": 0, "NA - Not a cardiovascular surgery": 0,
                               "First re-op cardiovascular surgery": 0, "Second re-op cardiovascular surgery": 1,
                               "Third re-op cardiovascular surgery": 1,
                               "Fourth or more re-op cardiovascular surgery": 1}})
        df = df.replace({col: {"Never smoker": 0, "Smoker": 1}})
        df = df.replace({col: {"I/II": 0, "III/IV": 1}})
        df = df.replace({col: {"None": 0, "One": 1, "Two": 2, "Three": 3}})
        # df_new_first = df_new_first.replace({col: {int("2"): 1, int("3"): 1, int("4"): 1, int("5"): 1, int("6"): 1}})
        # categories.append(col)

    except:
        x="none"
        # print(col)
# print(df)

categorical=['Gender','NumDisV', 'Ethnicity', 'FHCAD', 'RaceCaucasian', 'RaceBlack', 'RaceOther',
           'Hypertn', 'Dyslip', 'Diabetes', 'InsulinDiab', 'PVD', 'PrCVInt', 'SmokingStatus', 'ChrLungD',
           'ModSevereLungDis', 'RenFail', 'Dialysis', 'ImmSupp', 'InfEndo', 'PrevMI',
           'POCPCI', 'Angina', 'UnstableAngina', 'HeartFail', 'ClassNYHGroup', 'Arrhythmia', 'ArrhythAtrFibFlutter',
           'ArrhythOther', 'NumDisV_ordinal', 'LeftMain', 'VDStenA', 'VDInsufA',
           'VDStenM', 'VDInsufM', 'VDStenT', 'AntiCoag', 'MedASA', 'MedAntiplateltNoASA', 'DualAntiPlat', 'MedHeparin', 'MedNitIV', 'MedInotr',
           'MedACEI', 'MedBeta', 'MedSter']
print("hey")
df = pd.get_dummies(df, columns=categorical)

print(df.columns)
df_all=pd.DataFrame(df.columns.tolist())
df_all.to_csv("columns.csv")
formula = 'Reoperation ~ Age+Gender_0+Gender_1+RaceCaucasian_0+RaceCaucasian_1+RaceBlack_0+RaceBlack_1+RaceOther_0+RaceOther_1+Ethnicity_0+Ethnicity_1+BMI+FHCAD_0+FHCAD_1+Hypertn_0+Hypertn_1+Dyslip_0+Dyslip_1+Diabetes_0+Diabetes_1+InsulinDiab_0+InsulinDiab_1+PVD_0+PVD_1+PrCVInt_0+PrCVInt_1+SmokingStatus_0+SmokingStatus_1+ChrLungD_0+ChrLungD_1+ModSevereLungDis_0+ModSevereLungDis_1+RenFail_0+RenFail_1+Dialysis_0+Dialysis_1+ImmSupp_0+ImmSupp_1+InfEndo_0+InfEndo_1+PrevMI_0+PrevMI_1+POCPCI_0+POCPCI_1+Angina_0+Angina_1+UnstableAngina_0+UnstableAngina_1+HeartFail_0+HeartFail_1+ClassNYHGroup_0+ClassNYHGroup_1+Arrhythmia_0+Arrhythmia_1+ArrhythAtrFibFlutter_0+ArrhythAtrFibFlutter_1+ArrhythOther_0+ArrhythOther_1+NumDisV_1+NumDisV_2+NumDisV_3+LeftMain_1+VDStenA_0+VDStenA_1+VDInsufA_0+VDInsufA_1+VDStenM_0+VDStenM_1+VDInsufM_0+VDInsufM_1+VDStenT_0+VDStenT_1+NumDisV_ordinal_1+NumDisV_ordinal_2+NumDisV_ordinal_3+AntiCoag_0+AntiCoag_1+MedASA_0+MedASA_1+MedAntiplateltNoASA_0+MedAntiplateltNoASA_1+DualAntiPlat_0+DualAntiPlat_1+MedHeparin_0+MedHeparin_1+MedNitIV_0+MedNitIV_1+MedInotr_0+MedInotr_1+MedACEI_0+MedACEI_1+MedBeta_0+MedBeta_1+MedSter_0+MedSter_1'
model = smf.glm(formula = formula, data=df, family=sm.families.Binomial())
result = model.fit()
print(result.summary())
print('T-values: ', result.tvalues)
