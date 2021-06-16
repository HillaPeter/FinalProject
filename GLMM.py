import numpy as np
import pandas as pd
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM


def glmm_model(data, features, y, random_effects):
    model = BinomialBayesMixedGLM.from_formula(f'{y} ~ {features}', random_effects, data)
    result = model.fit_vb()
    return result


def call_glmm_model(data, features_basic, outcome_list, random_effects, list_of_featues_to_add, feature_number=0, outcome_number=0):
    features = features_basic + list_of_featues_to_add[feature_number]
    outcome = outcome_list[outcome_number]
    data['Mortalty'] = data['Mortalty'].astype(np.int64)
    return glmm_model(data, features, outcome, random_effects)


def add_Summary_Data_To_ImputedData(df):
    df1 = df.groupby(['HospID', 'surgyear'])['HospID'].count().reset_index(name='total_CABG')
    df2 = df.groupby(['HospID', 'surgyear'])['Reoperation'].apply(lambda x: (x == 'Reoperation').sum()).reset_index(
            name='Reop_CABG')
    df_aggr = pd.read_csv("aggregate_csv.csv")
    df3 = pd.merge(df1, df, left_on=['HospID', 'surgyear'], right_on=['HospID', 'surgyear'], how='outer')
    df4 = pd.merge(df2, df3, left_on=['HospID', 'surgyear'], right_on=['HospID', 'surgyear'], how='outer')
    df5 = pd.merge(df_aggr, df4, left_on=['HospID', 'surgyear'], right_on=['HospID', 'surgyear'],
                   how='inner')  # how='left', on=['HospID','surgyear'])
    del df5["Unnamed: 0"]
    df5.to_csv("imputed_data_with_sum.csv")
    print(df5.head(10))
    return df5


def refactor_categorical_values_to_numeric_values(df, col_names):
    # df = df.filter(col_names, axis=1)
    for col in col_names:
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
        except:
            x = "none"
    df.to_csv('/tmp/pycharm_project_957/imputed_data_with_float_values_glmm.csv')


def clean_data(df_to_clean):
    df_to_clean.rename(columns={"EF<=35%": "EF_less_equal_35"}, inplace=True)
    print("type of hosp id: ", type(df_to_clean['HospID_total_cardiac_surgery'][0]))

    df_to_clean['HospID_total_cardiac_surgery'] = df_to_clean['HospID_total_cardiac_surgery'].astype(str)
    df_to_clean['HospID_total_cardiac_surgery'] = df_to_clean['HospID_total_cardiac_surgery'].str.replace(',', '')
    df_to_clean['HospID_total_cardiac_surgery'] = df_to_clean['HospID_total_cardiac_surgery'].astype(np.float)
    df_to_clean.to_csv("imputed_data_after_cleaning_glmm.csv")


def create_reop_imputed_data(df_imputed_data):
    df_imputed_data = df_imputed_data[df_imputed_data['Reoperation'] == 'Reoperation']
    df_imputed_data.to_csv('/tmp/pycharm_project_957/imputed_data_reop.csv')


if __name__ == "__main__":
    df_reop = pd.read_csv('/tmp/pycharm_project_957/imputed_data_reop_clean.csv')
    df_reop = df_reop.dropna()
    df_reop.to_csv('/tmp/pycharm_project_957/imputed_data_reop_clean.csv')
    print()
    # yaara's script
    # df_all = pd.read_csv("/tmp/pycharm_project_957/imputed_data_sum_info_surg_and_Hosp.csv")
    # df_with_sum = add_Summary_Data_To_ImputedData(df_all)
    # df_columns = pd.DataFrame(df_all.columns)
    # df_columns.to_csv("columns_of_imputed_data.csv")

    # GLMM
    # read table
    # df_with_sum = pd.read_csv('/tmp/pycharm_project_957/imputed_data_with_sum.csv')
    # # change categorical values to numerical values
    # list_vals = ["surgyear", "Reoperation", "BMI", "Age", "Gender", "RaceCaucasian", "RaceBlack", "Ethnicity",
    #              "RaceOther", "FHCAD", "Diabetes", "InsulinDiab", "Dyslip", "Dialysis", "Hypertn", "InfEndo",
    #              "SmokingStatus", "ChrLungD", "ModSevereLungDis", "ImmSupp", "PVD", "DualAntiPlat", 'RenFail',
    #              "CreatLst", 'PreCVAorTIAorCVD', "POCPCI", "PrevMI", "Angina", "UnstableAngina", "HeartFail",
    #              "ClassNYHGroup", "Arrhythmia", "ArrhythAtrFibFlutter", "ArrhythOther", "MedACEI", "MedBeta",
    #              "MedNitIV", "MedASA", "MedAntiplateltNoASA", "AntiCoag", "MedInotr", "MedSter", "HDEF", "EF<=35%",
    #              "NumDisV", 'NumDisV_ordinal', "LeftMain", "VDInsufA", "VDStenA", "VDInsufM", "VDStenM", "VDInsufT",
    #              "VDStenT", "Status", 'MedHeparin', 'Mortality', 'PrCVInt']
    # # list_val = ['PrCVInt']
    # refactor_categorical_values_to_numeric_values(df_all, list_vals)

    df = pd.read_csv("/tmp/pycharm_project_957/imputed_data_after_cleaning_glmm.csv")
    df_short = df[:1000]

    features_basic = '''Age+surgyear+Reoperation+BMI+Gender+RaceCaucasian+RaceBlack+Ethnicity+RaceOther+FHCAD+Diabetes+
                InsulinDiab+Dyslip+Dialysis+Hypertn+InfEndo+SmokingStatus+ChrLungD+ModSevereLungDis+ImmSupp+PVD+DualAntiPlat+RenFail+
                CreatLst+PreCVAorTIAorCVD+POCPCI+PrevMI+Angina+UnstableAngina+HeartFail+ClassNYHGroup+Arrhythmia+ArrhythAtrFibFlutter+
                ArrhythOther+MedACEI+MedBeta+MedNitIV+MedASA+MedAntiplateltNoASA+AntiCoag+MedInotr+MedSter+HDEF+EF_less_equal_35+
                NumDisV+NumDisV_ordinal+LeftMain+VDInsufA+VDStenA+VDInsufM+VDStenM+VDInsufT+VDStenT+Status+MedHeparin+PrCVInt'''


    list_features_to_add = ['+HospID_total_CABG', '+HospID_total_cardiac_surgery', '+HospID_Reop_CABG']
    list_of_outcome = ['Mortalty', 'Complics']
    # features_CABG = features_basic + '+HospID_total_CABG'
    # features_total_surgeries = features_basic + '+HospID_total_cardiac_surgery'
    # features_reop = features_basic + '+HospID_Reop_CABG'
    # random_effect_variables = {'HospID': '0 + C(HospID)'}
    random_effect_variables = {'HospID': '0 + C(HospID)', 'surgid': '0 + C(surgid)'}

    result_glmm = call_glmm_model(df_short, features_basic, list_of_outcome, random_effect_variables, list_features_to_add)

    # result_glmm_total = glmm_model(df_with_sum, features_total_surgeries, 'Mortalty', random_effect_variables)
    # result_glmm_reop = glmm_model(df_with_sum, features_reop, 'Mortalty', random_effect_variables)

    print("result_glmm", result_glmm.summary())

