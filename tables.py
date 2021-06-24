from collections import OrderedDict

import pandas as pd
import scipy.stats


def bins(row, col):
    if row[col] < 100:
        val = 1
    elif 150 > row[col] >= 100:
        val = 2
    elif 200 > row[col] >= 150:
        val = 3
    elif 300 > row[col] >= 200:
        val = 4
    elif 300 <= row[col] <= 449:
        val = 5
    else:
        val = 6
    return val


def bins_reop(row, col):
    if row[col] < 10:
        val = 1
    elif 20 > row[col] >= 10:
        val = 2
    elif 30 > row[col] >= 20:
        val = 3
    else:
        val = 4
    return val


''' 
experience_hosp = Mortalty/Complics
hosp_or_surg = HospID/surgid
'''


def create_table(df, experience_hosp, hosp_or_surg):
    dict_number_of_hospitals_or_surgeries_in_each_bin = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
    dict_number_of_patient_in_each_bin = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
    dict_mortality_in_each_bin = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
    dict_mortality_first_in_each_bin = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
    dict_mortality_reop_in_each_bin = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
    total_hospitals_or_surgeries = len(df.drop_duplicates(subset=hosp_or_surg, keep='first'))

    for index, row in df.iterrows():
        if experience_hosp == 'Reop':
            bin_number = bins_reop(row, experience_hosp)
        else:
            bin_number = bins(row, experience_hosp)
        dict_number_of_patient_in_each_bin[bin_number] += row[experience_hosp]
        dict_number_of_hospitals_or_surgeries_in_each_bin[bin_number] += 1
        dict_mortality_in_each_bin[bin_number] += row['Complics_all']
        dict_mortality_reop_in_each_bin[bin_number] += row['Complics_reop']
        dict_mortality_first_in_each_bin[bin_number] += row['Complics_FirstOperation']
    total_patients = sum(dict_number_of_patient_in_each_bin.values())
    return total_hospitals_or_surgeries, total_patients, dict_number_of_hospitals_or_surgeries_in_each_bin, dict_number_of_patient_in_each_bin, dict_mortality_in_each_bin, dict_mortality_reop_in_each_bin, dict_mortality_first_in_each_bin


def get_stats(df, col):
    # mean_bin = df_for_one_bin[col].mean()  # mortality_rate
    values_of_col = df[col]
    median_bin = values_of_col.values.mean()  # mortality_rate
    bin_25 = values_of_col.describe()['25%']
    bin_75 = values_of_col.describe()['75%']
    std_bin = values_of_col.values.std()
    min_bin = df[col].min()  # mortality_rate
    max_bin = df[col].max()  # mortality_rate
    return round(median_bin, 3), round(bin_25, 3), round(bin_75, 3), round(min_bin, 3), round(max_bin, 3), std_bin


def statistic_test(df1, df2, df3, df4, df5, df6, column):
    if df6.empty:
        r, p = scipy.stats.kruskal(df1[column], df2[column], df3[column], df4[column], df5[column])
    else:
        r, p = scipy.stats.kruskal(df1[column], df2[column], df3[column], df4[column], df5[column], df6[column])
    # print(f'p-value of {column} is : {p}')
    return p


def statistic_test_reop(df1, df2, df3, df4, column):
    r, p = scipy.stats.kruskal(df1[column], df2[column], df3[column], df4[column])
    # print(f'p-value of {column} is : {p}')
    return p


def add_item_to_the_begin_of_list(first_item, list_items):
    whole_list = [first_item] + list_items
    return whole_list


def str_to_int(row, col):
    row[col] = (row[col]).replace(',', '')
    return int(float(row[col]))


def classify_each_row_to_bin_number(df_to_classify, func, col):
    return df_to_classify.apply(func, col=col, axis=1)


''' is_mort = True/False, columns_names = list of bins' ranges, exp = Hospitals/Surgeries,
 exp_col = CABG/total_cardiac_surgeries/Reop'''


def convert_dict_to_df(total_hospitals, total_patients, dict_number_of_hospitals_in_each_bin,
                       dict_number_of_patient_in_each_bin, mort_median, mort_first_median, mort_reop_median,
                       mort_minmax, mort_first_minmax, mort_reop_minmax, mort_std, mort_first_std, mort_reop_std,
                       mort_ove_median, mort_ove_first_median, mort_ove_reop_median,
                       mort_ove_minmax, mort_ove_first_minmax, mort_ove_reop_minmax, mort_ove_std, mort_ove_first_std,
                       mort_ove_reop_std, is_mort, columns_names, exp, exp_col, type_of_mort):
    if is_mort:
        outcome_col_name = 'Mortality'
    else:
        outcome_col_name = 'Complics'
    sorted_dict_hospitals = OrderedDict(sorted(dict_number_of_hospitals_in_each_bin.items(), key=lambda t: t[0]))
    sorted_dict_patients = OrderedDict(sorted(dict_number_of_patient_in_each_bin.items(), key=lambda t: t[0]))

    hospitals = [value for (key, value) in sorted_dict_hospitals.items()]
    hospitals.insert(0, total_hospitals)
    patients = [value for (key, value) in sorted_dict_patients.items()]
    patients.insert(0, total_patients)

    all_bins = [hospitals, patients, mort_median, mort_minmax, mort_std, mort_first_median, mort_first_minmax,
                mort_first_std,
                mort_reop_median, mort_reop_minmax, mort_reop_std, mort_ove_median, mort_ove_minmax, mort_ove_std,
                mort_ove_first_median, mort_ove_first_minmax, mort_ove_first_std,
                mort_ove_reop_median, mort_ove_reop_minmax, mort_ove_reop_std]

    # Create the pandas DataFrame
    # df_final = pd.DataFrame(all_bins,
    #                         columns=['All', '<100', '100-149', '150-199', '200-299', '300-449', '>=450'])
    df_final = pd.DataFrame(all_bins,
                            columns=columns_names)
    df_final.index = [exp, 'Patients', outcome_col_name, outcome_col_name + ' [25%,75%]',
                      outcome_col_name + ' Std', outcome_col_name + ' First',
                      outcome_col_name + ' First [25%,75%]', outcome_col_name + ' First Std',
                      outcome_col_name + ' Reop', outcome_col_name + ' Reop [25%,75%]', outcome_col_name + ' Reop Std',
                      outcome_col_name + ' OBS/EXP', outcome_col_name + ' OBS/EXP [25%,75%]',
                      outcome_col_name + ' OBS/EXP Std', outcome_col_name + ' OBS/EXP First',
                      outcome_col_name + ' OBS/EXP First [25%,75%]', outcome_col_name + ' OBS/EXP First Std',
                      outcome_col_name + ' OBS/EXP Reop', outcome_col_name + ' OBS/EXP Reop [25%,75%]',
                      outcome_col_name + ' OBS/EXP Reop Std']
    df_final.to_csv(f'{exp} Volume Category {outcome} - {exp_col} Volume {type_of_mort}.csv')
    return df_final


def info_of_bins_reop(outcome_col, experience_of, exp_col, type_of_mort):
    mask = df_all_years['bin_total'] == 1
    df1 = df_all_years[mask]
    mask = df_all_years['bin_total'] == 2
    df2 = df_all_years[mask]
    mask = df_all_years['bin_total'] == 3
    df3 = df_all_years[mask]
    mask = df_all_years['bin_total'] == 4
    df4 = df_all_years[mask]

    mean_bin1, bin1_25, bin1_75, min_bin1, max_bin1, std_bin1 = get_stats(df1, outcome_col + '_rate_All')
    mean_bin2, bin2_25, bin2_75, min_bin2, max_bin2, std_bin2 = get_stats(df2, outcome_col + '_rate_All')
    mean_bin3, bin3_25, bin3_75, min_bin3, max_bin3, std_bin3 = get_stats(df3, outcome_col + '_rate_All')
    mean_bin4, bin4_25, bin4_75, min_bin4, max_bin4, std_bin4 = get_stats(df4, outcome_col + '_rate_All')

    outcome_mean = [mean_bin1, mean_bin2, mean_bin3, mean_bin4]
    outcome_25_75 = [(bin1_25, bin1_75), (bin2_25, bin2_75), (bin3_25, bin3_75),
                     (bin4_25, bin4_75)]

    outcome_std = [std_bin1, std_bin2, std_bin3, std_bin4]

    # FIRST
    mean_first_bin1, bin1_first_25, bin1_first_75, min_first_bin1, max_first_bin1, std_first_bin1 = get_stats(df1,
                                                                                                              outcome_col + '_First_rate')
    mean_first_bin2, bin2_first_25, bin2_first_75, min_first_bin2, max_first_bin2, std_first_bin2 = get_stats(df2,
                                                                                                              outcome_col + '_First_rate')
    mean_first_bin3, bin3_first_25, bin3_first_75, min_first_bin3, max_first_bin3, std_first_bin3 = get_stats(df3,
                                                                                                              outcome_col + '_First_rate')
    mean_first_bin4, bin4_first_25, bin4_first_75, min_first_bin4, max_first_bin4, std_first_bin4 = get_stats(df4,
                                                                                                              outcome_col + '_First_rate')

    outcome_first_mean = [mean_first_bin1, mean_first_bin2, mean_first_bin3, mean_first_bin4]
    outcome_first_25_75 = [(bin1_first_25, bin1_first_75), (bin2_first_25, bin2_first_75),
                           (bin3_first_25, bin3_first_75), (bin4_first_25, bin4_first_75)]

    outcome_first_std = [std_first_bin1, std_first_bin2, std_first_bin3, std_first_bin4]
    # RE-OP
    # mask_hosp = df_all_years['Reop'] == 0
    # df_hosp_id_all_years_reop = df_all_years[~mask_hosp]

    mask = df1['Reop'] == 0
    df1_reop = df1[~mask]
    mask = df2['Reop'] == 0
    df2_reop = df2[~mask]
    mask = df3['Reop'] == 0
    df3_reop = df3[~mask]
    mask = df4['Reop'] == 0
    df4_reop = df4[~mask]

    mean_reop_bin1, bin1_reop_25, bin1_reop_75, min_reop_bin1, max_reop_bin1, std_reop_bin1 = get_stats(df1_reop,
                                                                                                        outcome_col + '_Reop_rate')
    mean_reop_bin2, bin2_reop_25, bin2_reop_75, min_reop_bin2, max_reop_bin2, std_reop_bin2 = get_stats(df2_reop,
                                                                                                        outcome_col + '_Reop_rate')
    mean_reop_bin3, bin3_reop_25, bin3_reop_75, min_reop_bin3, max_reop_bin3, std_reop_bin3 = get_stats(df3_reop,
                                                                                                        outcome_col + '_Reop_rate')
    mean_reop_bin4, bin4_reop_25, bin4_reop_75, min_reop_bin4, max_reop_bin4, std_reop_bin4 = get_stats(df4_reop,
                                                                                                        outcome_col + '_Reop_rate')

    outcome_reop_mean = [mean_reop_bin1, mean_reop_bin2, mean_reop_bin3, mean_reop_bin4]
    outcome_reop_25_75 = [(bin1_reop_25, bin1_reop_75), (bin2_reop_25, bin2_reop_75), (bin3_reop_25, bin3_reop_75),
                          (bin4_reop_25, bin4_reop_75)]

    outcome_reop_std = [std_reop_bin1, std_reop_bin2, std_reop_bin3, std_reop_bin4]

    # OBS/EXP
    mean_ove_bin1, bin1_ove_25, bin1_ove_75, min_ove_bin1, max_ove_bin1, std_ove_bin1 = get_stats(df1,
                                                                                                  outcome_col + '_observe/expected_All')
    mean_ove_bin2, bin2_ove_25, bin2_ove_75, min_ove_bin2, max_ove_bin2, std_ove_bin2 = get_stats(df2,
                                                                                                  outcome_col + '_observe/expected_All')
    mean_ove_bin3, bin3_ove_25, bin3_ove_75, min_ove_bin3, max_ove_bin3, std_ove_bin3 = get_stats(df3,
                                                                                                  outcome_col + '_observe/expected_All')
    mean_ove_bin4, bin4_ove_25, bin4_ove_75, min_ove_bin4, max_ove_bin4, std_ove_bin4 = get_stats(df4,
                                                                                                  outcome_col + '_observe/expected_All')

    outcome_ove_mean = [mean_ove_bin1, mean_ove_bin2, mean_ove_bin3, mean_ove_bin4]
    outcome_ove_25_75 = [(bin1_ove_25, bin1_ove_75), (bin2_ove_25, bin2_ove_75), (bin3_ove_25, bin3_ove_75),
                         (bin4_ove_25, bin4_ove_75)]

    outcome_ove_std = [std_ove_bin1, std_ove_bin2, std_ove_bin3, std_ove_bin4]

    # OBS/EXP First

    mask = df1[outcome_col + '_observe/expected_First'] == 0
    df1_first_ove = df1[~mask]
    mask = df2[outcome_col + '_observe/expected_First'] == 0
    df2_first_ove = df2[~mask]
    mask = df3[outcome_col + '_observe/expected_First'] == 0
    df3_first_ove = df3[~mask]
    mask = df4[outcome_col + '_observe/expected_First'] == 0
    df4_first_ove = df4[~mask]

    mean_ove_first_bin1, bin1_ove_first_25, bin1_ove_first_75, min_ove_first_bin1, max_ove_first_bin1, std_ove_first_bin1 = get_stats(
        df1_first_ove, outcome_col + '_observe/expected_First')
    mean_ove_first_bin2, bin2_ove_first_25, bin2_ove_first_75, min_ove_first_bin2, max_ove_first_bin2, std_ove_first_bin2 = get_stats(
        df2_first_ove, outcome_col + '_observe/expected_First')
    mean_ove_first_bin3, bin3_ove_first_25, bin3_ove_first_75, min_ove_first_bin3, max_ove_first_bin3, std_ove_first_bin3 = get_stats(
        df3_first_ove, outcome_col + '_observe/expected_First')
    mean_ove_first_bin4, bin4_ove_first_25, bin4_ove_first_75, min_ove_first_bin4, max_ove_first_bin4, std_ove_first_bin4 = get_stats(
        df4_first_ove, outcome_col + '_observe/expected_First')

    outcome_ove_first_mean = [mean_ove_first_bin1, mean_ove_first_bin2, mean_ove_first_bin3,
                              mean_ove_first_bin4]
    outcome_ove_first_25_75 = [(bin1_ove_first_25, bin1_ove_first_75), (bin2_ove_first_25, bin2_ove_first_75),
                               (bin3_ove_first_25, bin3_ove_first_75), (bin4_ove_first_25, bin4_ove_first_75)]

    outcome_ove_first_std = [std_ove_first_bin1, std_ove_first_bin2, std_ove_first_bin3, std_ove_first_bin4]

    # OBS/EXP RE-OP
    mask = df1_reop[outcome_col + '_observe/expected_Reop'] == 0
    df1_reop_ove = df1_reop[~mask]
    mask = df2_reop[outcome_col + '_observe/expected_Reop'] == 0
    df2_reop_ove = df2_reop[~mask]
    mask = df3_reop[outcome_col + '_observe/expected_Reop'] == 0
    df3_reop_ove = df3_reop[~mask]
    mask = df4_reop[outcome_col + '_observe/expected_Reop'] == 0
    df4_reop_ove = df4_reop[~mask]

    mean_ove_reop_bin1, bin1_ove_reop_25, bin1_ove_reop_75, min_ove_reop_bin1, max_ove_reop_bin1, std_ove_reop_bin1 = get_stats(
        df1_reop_ove, outcome_col + '_observe/expected_Reop')
    mean_ove_reop_bin2, bin2_ove_reop_25, bin2_ove_reop_75, min_ove_reop_bin2, max_ove_reop_bin2, std_ove_reop_bin2 = get_stats(
        df2_reop_ove, outcome_col + '_observe/expected_Reop')
    mean_ove_reop_bin3, bin3_ove_reop_25, bin3_ove_reop_75, min_ove_reop_bin3, max_ove_reop_bin3, std_ove_reop_bin3 = get_stats(
        df3_reop_ove, outcome_col + '_observe/expected_Reop')
    mean_ove_reop_bin4, bin4_ove_reop_25, bin4_ove_reop_75, min_ove_reop_bin4, max_ove_reop_bin4, std_ove_reop_bin4 = get_stats(
        df4_reop_ove, outcome_col + '_observe/expected_Reop')

    outcome_ove_reop_mean = [mean_ove_reop_bin1, mean_ove_reop_bin2, mean_ove_reop_bin3, mean_ove_reop_bin4]
    outcome_ove_reop_25_75 = [(bin1_ove_reop_25, bin1_ove_reop_75), (bin2_ove_reop_25, bin2_ove_reop_75),
                              (bin3_ove_reop_25, bin3_ove_reop_75), (bin4_ove_reop_25, bin4_ove_reop_75)]

    outcome_ove_reop_std = [std_ove_reop_bin1, std_ove_reop_bin2, std_ove_reop_bin3, std_ove_reop_bin4]

    # STATISTIC TEST
    p_outcome_all = statistic_test_reop(df1, df2, df3, df4, outcome_col + '_rate_All')
    p_outcome_first = statistic_test_reop(df1, df2, df3, df4, outcome_col + '_First_rate')
    p_outcome_reop = statistic_test_reop(df1_reop, df2_reop, df3_reop, df4_reop, outcome_col + '_Reop_rate')
    p_outcome_ove = statistic_test_reop(df1, df2, df3, df4, outcome_col + '_observe/expected_All')
    p_outcome_ove_first = statistic_test_reop(df1, df2, df3, df4, outcome_col + '_observe/expected_First')
    p_outcome_ove_reop = statistic_test_reop(df1_reop, df2_reop, df3_reop, df4_reop,
                                             outcome_col + '_observe/expected_Reop')

    outcome_mean.append(p_outcome_all)
    outcome_first_mean.append(p_outcome_first)
    outcome_reop_mean.append(p_outcome_reop)
    outcome_ove_mean.append(p_outcome_ove)
    outcome_ove_first_mean.append(p_outcome_ove_first)
    outcome_ove_reop_mean.append(p_outcome_ove_reop)
    outcome_mean = add_item_to_the_begin_of_list(outcome_median_all, outcome_mean)
    outcome_first_mean = add_item_to_the_begin_of_list(outcome_first_mean_all, outcome_first_mean)
    outcome_reop_mean = add_item_to_the_begin_of_list(outcome_reop_mean_all, outcome_reop_mean)
    outcome_25_75 = add_item_to_the_begin_of_list((outcome_25_all, outcome_75_all), outcome_25_75)
    outcome_first_25_75 = add_item_to_the_begin_of_list((outcome_first_25_all, outcome_first_75_all),
                                                        outcome_first_25_75)
    outcome_reop_25_75 = add_item_to_the_begin_of_list((outcome_reop_25_all, outcome_reop_75_all),
                                                       outcome_reop_25_75)
    outcome_std = [outcome_std_all] + outcome_std
    outcome_first_std = [outcome_first_std_all] + outcome_first_std
    outcome_reop_std = [outcome_reop_std_all] + outcome_reop_std
    outcome_ove_mean = add_item_to_the_begin_of_list(outcome_ove_mean_all,
                                                     outcome_ove_mean)
    outcome_ove_first_mean = add_item_to_the_begin_of_list(outcome_ove_first_median_all, outcome_ove_first_mean)
    outcome_ove_reop_mean = add_item_to_the_begin_of_list(outcome_ove_reop_mean_all, outcome_ove_reop_mean)
    outcome_ove_25_75 = add_item_to_the_begin_of_list((outcome_ove_all_25, outcome_ove_all_75),
                                                      outcome_ove_25_75)  # obs vs exp 25-75
    outcome_ove_first_25_75 = add_item_to_the_begin_of_list((outcome_ove_first_all_25, outcome_ove_first_all_75),
                                                            outcome_ove_first_25_75)
    outcome_ove_reop_25_75 = add_item_to_the_begin_of_list((outcome_ove_reop_all_25, outcome_ove_reop_all_75),
                                                           outcome_ove_reop_25_75)
    outcome_ove_std = add_item_to_the_begin_of_list(outcome_ove_std_all, outcome_ove_std)  # obs vs exp std
    outcome_ove_first_std = add_item_to_the_begin_of_list(outcome_ove_first_std_all, outcome_ove_first_std)
    outcome_ove_reop_std = add_item_to_the_begin_of_list(outcome_ove_reop_std_all, outcome_ove_reop_std)

    if outcome_col == 'Mortalty':
        mort = True
    else:
        mort = False
    columns_names = ['All', '<10', '10-20', '20-30', '>30', 'P-Value']

    del dict_number_of_hospitals_in_each_bin[6]
    del dict_number_of_hospitals_in_each_bin[5]
    del dict_number_of_patient_in_each_bin[6]
    del dict_number_of_patient_in_each_bin[5]

    # COMBINE ALL TOGETHER
    df_All = convert_dict_to_df(total_hospitals_or_surgeries, total_patients, dict_number_of_hospitals_in_each_bin,
                                dict_number_of_patient_in_each_bin, outcome_mean, outcome_first_mean, outcome_reop_mean,
                                outcome_25_75, outcome_first_25_75, outcome_reop_25_75, outcome_std, outcome_first_std,
                                outcome_reop_std,
                                outcome_ove_mean, outcome_ove_first_mean, outcome_ove_reop_mean,
                                outcome_ove_25_75, outcome_ove_first_25_75, outcome_ove_reop_25_75, outcome_ove_std,
                                outcome_ove_first_std,
                                outcome_ove_reop_std, mort, columns_names, experience_of, exp_col, type_of_mort)


def info_of_bins(outcome_col, experience_of, exp_col, type_of_mort):
    mask = df_all_years['bin_total'] == 1
    df1 = df_all_years[mask]
    mask = df_all_years['bin_total'] == 2
    df2 = df_all_years[mask]
    mask = df_all_years['bin_total'] == 3
    df3 = df_all_years[mask]
    mask = df_all_years['bin_total'] == 4
    df4 = df_all_years[mask]
    mask = df_all_years['bin_total'] == 5
    df5 = df_all_years[mask]
    mask = df_all_years['bin_total'] == 6
    df6 = df_all_years[mask]

    mean_bin1, bin1_25, bin1_75, min_bin1, max_bin1, std_bin1 = get_stats(df1, outcome_col + '_rate_All')
    mean_bin2, bin2_25, bin2_75, min_bin2, max_bin2, std_bin2 = get_stats(df2, outcome_col + '_rate_All')
    mean_bin3, bin3_25, bin3_75, min_bin3, max_bin3, std_bin3 = get_stats(df3, outcome_col + '_rate_All')
    mean_bin4, bin4_25, bin4_75, min_bin4, max_bin4, std_bin4 = get_stats(df4, outcome_col + '_rate_All')
    mean_bin5, bin5_25, bin5_75, min_bin5, max_bin5, std_bin5 = get_stats(df5, outcome_col + '_rate_All')
    mean_bin6, bin6_25, bin6_75, min_bin6, max_bin6, std_bin6 = get_stats(df6, outcome_col + '_rate_All')

    outcome_mean = [mean_bin1, mean_bin2, mean_bin3, mean_bin4, mean_bin5, mean_bin6]
    outcome_25_75 = [(bin1_25, bin1_75), (bin2_25, bin2_75), (bin3_25, bin3_75),
                     (bin4_25, bin4_75), (bin5_25, bin5_75), (bin6_25, bin6_75)]

    outcome_std = [std_bin1, std_bin2, std_bin3, std_bin4, std_bin5, std_bin6]

    # FIRST
    mean_first_bin1, bin1_first_25, bin1_first_75, min_first_bin1, max_first_bin1, std_first_bin1 = get_stats(df1,
                                                                                                              outcome_col + '_First_rate')
    mean_first_bin2, bin2_first_25, bin2_first_75, min_first_bin2, max_first_bin2, std_first_bin2 = get_stats(df2,
                                                                                                              outcome_col + '_First_rate')
    mean_first_bin3, bin3_first_25, bin3_first_75, min_first_bin3, max_first_bin3, std_first_bin3 = get_stats(df3,
                                                                                                              outcome_col + '_First_rate')
    mean_first_bin4, bin4_first_25, bin4_first_75, min_first_bin4, max_first_bin4, std_first_bin4 = get_stats(df4,
                                                                                                              outcome_col + '_First_rate')
    mean_first_bin5, bin5_first_25, bin5_first_75, min_first_bin5, max_first_bin5, std_first_bin5 = get_stats(df5,
                                                                                                              outcome_col + '_First_rate')
    mean_first_bin6, bin6_first_25, bin6_first_75, min_first_bin6, max_first_bin6, std_first_bin6 = get_stats(df6,
                                                                                                              outcome_col + '_First_rate')

    outcome_first_mean = [mean_first_bin1, mean_first_bin2, mean_first_bin3, mean_first_bin4,
                          mean_first_bin5,
                          mean_first_bin6]
    outcome_first_25_75 = [(bin1_first_25, bin1_first_75), (bin2_first_25, bin2_first_75),
                           (bin3_first_25, bin3_first_75),
                           (bin4_first_25, bin4_first_75), (bin5_first_25, bin5_first_75),
                           (bin6_first_25, bin6_first_75)]

    outcome_first_std = [std_first_bin1, std_first_bin2, std_first_bin3, std_first_bin4, std_first_bin5,
                         std_first_bin6]

    # RE-OP
    # mask_hosp = df_all_years[exp_col] == 0
    # df_hosp_id_all_years_reop = df_all_years[~mask_hosp]

    mask = df1['Reop'] == 0
    df1_reop = df1[~mask]
    mask = df2['Reop'] == 0
    df2_reop = df2[~mask]
    mask = df3['Reop'] == 0
    df3_reop = df3[~mask]
    mask = df4['Reop'] == 0
    df4_reop = df4[~mask]
    mask = df5['Reop'] == 0
    df5_reop = df5[~mask]
    mask = df6['Reop'] == 0
    df6_reop = df6[~mask]

    mean_reop_bin1, bin1_reop_25, bin1_reop_75, min_reop_bin1, max_reop_bin1, std_reop_bin1 = get_stats(df1_reop,
                                                                                                        outcome_col + '_Reop_rate')
    mean_reop_bin2, bin2_reop_25, bin2_reop_75, min_reop_bin2, max_reop_bin2, std_reop_bin2 = get_stats(df2_reop,
                                                                                                        outcome_col + '_Reop_rate')
    mean_reop_bin3, bin3_reop_25, bin3_reop_75, min_reop_bin3, max_reop_bin3, std_reop_bin3 = get_stats(df3_reop,
                                                                                                        outcome_col + '_Reop_rate')
    mean_reop_bin4, bin4_reop_25, bin4_reop_75, min_reop_bin4, max_reop_bin4, std_reop_bin4 = get_stats(df4_reop,
                                                                                                        outcome_col + '_Reop_rate')
    mean_reop_bin5, bin5_reop_25, bin5_reop_75, min_reop_bin5, max_reop_bin5, std_reop_bin5 = get_stats(df5_reop,
                                                                                                        outcome_col + '_Reop_rate')
    mean_reop_bin6, bin6_reop_25, bin6_reop_75, min_reop_bin6, max_reop_bin6, std_reop_bin6 = get_stats(df6_reop,
                                                                                                        outcome_col + '_Reop_rate')

    outcome_reop_mean = [mean_reop_bin1, mean_reop_bin2, mean_reop_bin3, mean_reop_bin4, mean_reop_bin5,
                         mean_reop_bin6]
    outcome_reop_25_75 = [(bin1_reop_25, bin1_reop_75), (bin2_reop_25, bin2_reop_75), (bin3_reop_25, bin3_reop_75),
                          (bin4_reop_25, bin4_reop_75), (bin5_reop_25, bin5_reop_75), (bin6_reop_25, bin6_reop_75)]

    outcome_reop_std = [std_reop_bin1, std_reop_bin2, std_reop_bin3, std_reop_bin4, std_reop_bin5, std_reop_bin6]

    # OBS/EXP
    mean_ove_bin1, bin1_ove_25, bin1_ove_75, min_ove_bin1, max_ove_bin1, std_ove_bin1 = get_stats(df1,
                                                                                                  outcome_col + '_observe/expected_All')
    mean_ove_bin2, bin2_ove_25, bin2_ove_75, min_ove_bin2, max_ove_bin2, std_ove_bin2 = get_stats(df2,
                                                                                                  outcome_col + '_observe/expected_All')
    mean_ove_bin3, bin3_ove_25, bin3_ove_75, min_ove_bin3, max_ove_bin3, std_ove_bin3 = get_stats(df3,
                                                                                                  outcome_col + '_observe/expected_All')
    mean_ove_bin4, bin4_ove_25, bin4_ove_75, min_ove_bin4, max_ove_bin4, std_ove_bin4 = get_stats(df4,
                                                                                                  outcome_col + '_observe/expected_All')
    mean_ove_bin5, bin5_ove_25, bin5_ove_75, min_ove_bin5, max_ove_bin5, std_ove_bin5 = get_stats(df5,
                                                                                                  outcome_col + '_observe/expected_All')
    mean_ove_bin6, bin6_ove_25, bin6_ove_75, min_ove_bin6, max_ove_bin6, std_ove_bin6 = get_stats(df6,
                                                                                                  outcome_col + '_observe/expected_All')

    outcome_ove_mean = [mean_ove_bin1, mean_ove_bin2, mean_ove_bin3, mean_ove_bin4, mean_ove_bin5,
                        mean_ove_bin6]
    outcome_ove_25_75 = [(bin1_ove_25, bin1_ove_75), (bin2_ove_25, bin2_ove_75), (bin3_ove_25, bin3_ove_75),
                         (bin4_ove_25, bin4_ove_75), (bin5_ove_25, bin5_ove_75), (bin6_ove_25, bin6_ove_75)]

    outcome_ove_std = [std_ove_bin1, std_ove_bin2, std_ove_bin3, std_ove_bin4, std_ove_bin5, std_ove_bin6]

    # OBS/EXP First

    mask = df1[outcome_col + '_observe/expected_First'] == 0
    df1_first_ove = df1[~mask]
    mask = df2[outcome_col + '_observe/expected_First'] == 0
    df2_first_ove = df2[~mask]
    mask = df3[outcome_col + '_observe/expected_First'] == 0
    df3_first_ove = df3[~mask]
    mask = df4[outcome_col + '_observe/expected_First'] == 0
    df4_first_ove = df4[~mask]
    mask = df5[outcome_col + '_observe/expected_First'] == 0
    df5_first_ove = df5[~mask]
    mask = df6[outcome_col + '_observe/expected_First'] == 0
    df6_first_ove = df6[~mask]

    mean_ove_first_bin1, bin1_ove_first_25, bin1_ove_first_75, min_ove_first_bin1, max_ove_first_bin1, std_ove_first_bin1 = get_stats(
        df1_first_ove, outcome_col + '_observe/expected_First')
    mean_ove_first_bin2, bin2_ove_first_25, bin2_ove_first_75, min_ove_first_bin2, max_ove_first_bin2, std_ove_first_bin2 = get_stats(
        df2_first_ove, outcome_col + '_observe/expected_First')
    mean_ove_first_bin3, bin3_ove_first_25, bin3_ove_first_75, min_ove_first_bin3, max_ove_first_bin3, std_ove_first_bin3 = get_stats(
        df3_first_ove, outcome_col + '_observe/expected_First')
    mean_ove_first_bin4, bin4_ove_first_25, bin4_ove_first_75, min_ove_first_bin4, max_ove_first_bin4, std_ove_first_bin4 = get_stats(
        df4_first_ove, outcome_col + '_observe/expected_First')
    mean_ove_first_bin5, bin5_ove_first_25, bin5_ove_first_75, min_ove_first_bin5, max_ove_first_bin5, std_ove_first_bin5 = get_stats(
        df5_first_ove, outcome_col + '_observe/expected_First')
    mean_ove_first_bin6, bin6_ove_first_25, bin6_ove_first_75, min_ove_first_bin6, max_ove_first_bin6, std_ove_first_bin6 = get_stats(
        df6_first_ove, outcome_col + '_observe/expected_First')

    outcome_ove_first_mean = [mean_ove_first_bin1, mean_ove_first_bin2, mean_ove_first_bin3,
                              mean_ove_first_bin4, mean_ove_first_bin5, mean_ove_first_bin6]
    outcome_ove_first_25_75 = [(bin1_ove_first_25, bin1_ove_first_75), (bin2_ove_first_25, bin2_ove_first_75),
                               (bin3_ove_first_25, bin3_ove_first_75),
                               (bin4_ove_first_25, bin4_ove_first_75), (bin5_ove_first_25, bin5_ove_first_75),
                               (bin6_ove_first_25, bin6_ove_first_75)]

    outcome_ove_first_std = [std_ove_first_bin1, std_ove_first_bin2, std_ove_first_bin3, std_ove_first_bin4,
                             std_ove_first_bin5, std_ove_first_bin6]

    # OBS/EXP RE-OP
    mask = df1_reop[outcome_col + '_observe/expected_Reop'] == 0
    df1_reop_ove = df1_reop[~mask]
    mask = df2_reop[outcome_col + '_observe/expected_Reop'] == 0
    df2_reop_ove = df2_reop[~mask]
    mask = df3_reop[outcome_col + '_observe/expected_Reop'] == 0
    df3_reop_ove = df3_reop[~mask]
    mask = df4_reop[outcome_col + '_observe/expected_Reop'] == 0
    df4_reop_ove = df4_reop[~mask]
    mask = df5_reop[outcome_col + '_observe/expected_Reop'] == 0
    df5_reop_ove = df5_reop[~mask]
    mask = df6_reop[outcome_col + '_observe/expected_Reop'] == 0
    df6_reop_ove = df6_reop[~mask]

    mean_ove_reop_bin1, bin1_ove_reop_25, bin1_ove_reop_75, min_ove_reop_bin1, max_ove_reop_bin1, std_ove_reop_bin1 = get_stats(
        df1_reop_ove, outcome_col + '_observe/expected_Reop')
    mean_ove_reop_bin2, bin2_ove_reop_25, bin2_ove_reop_75, min_ove_reop_bin2, max_ove_reop_bin2, std_ove_reop_bin2 = get_stats(
        df2_reop_ove, outcome_col + '_observe/expected_Reop')
    mean_ove_reop_bin3, bin3_ove_reop_25, bin3_ove_reop_75, min_ove_reop_bin3, max_ove_reop_bin3, std_ove_reop_bin3 = get_stats(
        df3_reop_ove, outcome_col + '_observe/expected_Reop')
    mean_ove_reop_bin4, bin4_ove_reop_25, bin4_ove_reop_75, min_ove_reop_bin4, max_ove_reop_bin4, std_ove_reop_bin4 = get_stats(
        df4_reop_ove, outcome_col + '_observe/expected_Reop')
    mean_ove_reop_bin5, bin5_ove_reop_25, bin5_ove_reop_75, min_ove_reop_bin5, max_ove_reop_bin5, std_ove_reop_bin5 = get_stats(
        df5_reop_ove, outcome_col + '_observe/expected_Reop')
    mean_ove_reop_bin6, bin6_ove_reop_25, bin6_ove_reop_75, min_ove_reop_bin6, max_ove_reop_bin6, std_ove_reop_bin6 = get_stats(
        df6_reop_ove, outcome_col + '_observe/expected_Reop')

    outcome_ove_reop_mean = [mean_ove_reop_bin1, mean_ove_reop_bin2, mean_ove_reop_bin3, mean_ove_reop_bin4,
                             mean_ove_reop_bin5, mean_ove_reop_bin6]
    outcome_ove_reop_25_75 = [(bin1_ove_reop_25, bin1_ove_reop_75), (bin2_ove_reop_25, bin2_ove_reop_75),
                              (bin3_ove_reop_25, bin3_ove_reop_75),
                              (bin4_ove_reop_25, bin4_ove_reop_75), (bin5_ove_reop_25, bin5_ove_reop_75),
                              (bin6_ove_reop_25, bin6_ove_reop_75)]

    outcome_ove_reop_std = [std_ove_reop_bin1, std_ove_reop_bin2, std_ove_reop_bin3, std_ove_reop_bin4,
                            std_ove_reop_bin5, std_ove_reop_bin6]

    # STATISTIC TEST
    p_outcome_all = statistic_test(df1, df2, df3, df4, df5, df6, outcome_col + '_rate_All')
    p_outcome_first = statistic_test(df1, df2, df3, df4, df5, df6, outcome_col + '_First_rate')
    p_outcome_reop = statistic_test(df1_reop, df2_reop, df3_reop, df4_reop, df5_reop, df6_reop,
                                    outcome_col + '_Reop_rate')
    p_outcome_ove = statistic_test(df1, df2, df3, df4, df5, df6, outcome_col + '_observe/expected_All')
    p_outcome_ove_first = statistic_test(df1, df2, df3, df4, df5, df6, outcome_col + '_observe/expected_First')
    p_outcome_ove_reop = statistic_test(df1_reop, df2_reop, df3_reop, df4_reop, df5_reop, df6_reop,
                                        outcome_col + '_observe/expected_Reop')

    outcome_mean.append(p_outcome_all)
    outcome_first_mean.append(p_outcome_first)
    outcome_reop_mean.append(p_outcome_reop)
    outcome_ove_mean.append(p_outcome_ove)
    outcome_ove_first_mean.append(p_outcome_ove_first)
    outcome_ove_reop_mean.append(p_outcome_ove_reop)

    outcome_mean = add_item_to_the_begin_of_list(outcome_median_all, outcome_mean)
    outcome_first_mean = add_item_to_the_begin_of_list(outcome_first_mean_all, outcome_first_mean)
    outcome_reop_mean = add_item_to_the_begin_of_list(outcome_reop_mean_all, outcome_reop_mean)
    outcome_25_75 = add_item_to_the_begin_of_list((outcome_25_all, outcome_75_all), outcome_25_75)
    outcome_first_25_75 = add_item_to_the_begin_of_list((outcome_first_25_all, outcome_first_75_all),
                                                        outcome_first_25_75)
    outcome_reop_25_75 = add_item_to_the_begin_of_list((outcome_reop_25_all, outcome_reop_75_all),
                                                       outcome_reop_25_75)
    outcome_std = [outcome_std_all] + outcome_std
    outcome_first_std = [outcome_first_std_all] + outcome_first_std
    outcome_reop_std = [outcome_reop_std_all] + outcome_reop_std

    outcome_ove_mean = add_item_to_the_begin_of_list(outcome_ove_mean_all,
                                                     outcome_ove_mean)
    outcome_ove_first_mean = add_item_to_the_begin_of_list(outcome_ove_first_median_all, outcome_ove_first_mean)
    outcome_ove_reop_mean = add_item_to_the_begin_of_list(outcome_ove_reop_mean_all, outcome_ove_reop_mean)
    outcome_ove_25_75 = add_item_to_the_begin_of_list((outcome_ove_all_25, outcome_ove_all_75),
                                                      outcome_ove_25_75)  # obs vs exp 25-75
    outcome_ove_first_25_75 = add_item_to_the_begin_of_list((outcome_ove_first_all_25, outcome_ove_first_all_75),
                                                            outcome_ove_first_25_75)
    outcome_ove_reop_25_75 = add_item_to_the_begin_of_list((outcome_ove_reop_all_25, outcome_ove_reop_all_75),
                                                           outcome_ove_reop_25_75)
    outcome_ove_std = add_item_to_the_begin_of_list(outcome_ove_std_all, outcome_ove_std)  # obs vs exp std
    outcome_ove_first_std = add_item_to_the_begin_of_list(outcome_ove_first_std_all, outcome_ove_first_std)
    outcome_ove_reop_std = add_item_to_the_begin_of_list(outcome_ove_reop_std_all, outcome_ove_reop_std)

    if outcome_col == 'Mortalty':
        mort = True
    else:
        mort = False
    columns_names = ['All', '<100', '100-149', '150-199', '200-299', '300-449', '>=450', 'P-Value']
    # COMBINE ALL TOGETHER
    df_All = convert_dict_to_df(total_hospitals_or_surgeries, total_patients, dict_number_of_hospitals_in_each_bin,
                                dict_number_of_patient_in_each_bin, outcome_mean, outcome_first_mean, outcome_reop_mean,
                                outcome_25_75, outcome_first_25_75, outcome_reop_25_75, outcome_std, outcome_first_std,
                                outcome_reop_std,
                                outcome_ove_mean, outcome_ove_first_mean, outcome_ove_reop_mean,
                                outcome_ove_25_75, outcome_ove_first_25_75, outcome_ove_reop_25_75, outcome_ove_std,
                                outcome_ove_first_std,
                                outcome_ove_reop_std, mort, columns_names, experience_of, exp_col, type_of_mort)


def rename_columns_names(df_to_rename, path):
    df_to_rename.rename(columns={"total surgery count": "total_surgery_count", "total": "total_CABG",
                                 "Mortality_all": "Mortalty_all", "mort_rate_All": "Mortalty_rate_All",
                                 "Mortality_First_rate": "Mortalty_First_rate",
                                 "Mortality_Reop_rate": "Mortalty_Reop_rate",
                                 "Mort_observe/expected_All": "Mortalty_observe/expected_All",
                                 "Mortalty_observe/expected_First": "",
                                 "Mort_observe/expected_Reop": "Mortalty_observe/expected_Reop",
                                 "Mort_observe/expected_Reop": "Mortalty_observe/expected_Reop",
                                 "Comp_observe/expected_All": "Complics_observe/expected_All",
                                 "Comp_observe/expected_First": "Complics_observe/expected_First",
                                 "Comp_observe/expected_Reop": "Complics_observe/expected_Reop"})
    df_to_rename.to_csv(path)


'''
path_str: the path of the file
return - hosp_or_surg : "surgid" or "HospID"
         type : STSRCOM, STRCHospD, STSRCMM
'''


def get_types(path_str):
    arr_path = path_str.split("/")
    splitted_name_of_file = arr_path[-1].split("_")
    hosp_or_surg = splitted_name_of_file[0]
    type = splitted_name_of_file[-1].split('.')[0]
    return hosp_or_surg, type


if __name__ == "__main__":
    path = f'/tmp/pycharm_project_957/surgid_allyears_expec_surgid_STSRCOM.csv'
    df_all_years = pd.read_csv(path)
    # rename_columns_names(df_all_years, path)

    hosp_or_surge, type_of_mortality = get_types(path)
    # df_all_years['total_surgery_count'] = df_all_years.apply(str_to_int, col='total_surgery_count', axis=1) # convert all col values to int
    # df_all_years.to_csv('/tmp/pycharm_project_957/hospid_allyears_expec_hospid_stsrcom.csv')
    '''create 'bin total' col and classify each row to a bin number'''
    list_of_col_names_for_classify_to_bins = ['total_surgery_count', 'total_CABG', 'Reop']
    list_of_col_names_for_outcome = ['Mortalty', 'Complics']
    for exp in list_of_col_names_for_classify_to_bins:
        for outcome in list_of_col_names_for_outcome:
            total_hospitals_or_surgeries, total_patients, dict_number_of_hospitals_in_each_bin, dict_number_of_patient_in_each_bin, \
            dict_outcome_in_each_bin, dict_outcome_reop_in_each_bin, dict_outcome_first_in_each_bin = create_table(
                df_all_years, exp, hosp_or_surge)
            # ALL
            outcome_median_all, outcome_all_25, outcome_all_75, outcome_25_all, outcome_75_all, outcome_std_all = \
                get_stats(df_all_years, outcome + '_rate_All')

            outcome_first_mean_all, outcome_first_all_25, outcome_first_all_75, outcome_first_25_all, \
            outcome_first_75_all, outcome_first_std_all = get_stats(df_all_years, outcome + '_First_rate')

            mask_hosp = df_all_years['Reop'] == 0
            df_hosp_id_all_years_reop = df_all_years[~mask_hosp]

            outcome_reop_mean_all, outcome_reop_all_25, outcome_reop_all_75, outcome_reop_25_all, \
            outcome_reop_75_all, outcome_reop_std_all = get_stats(df_hosp_id_all_years_reop, outcome + '_Reop_rate')

            # Obs vs exp
            mask_ove_all = df_all_years[outcome + '_observe/expected_All'] == 0
            df_hosp_id_all_years_ove_all = df_all_years[~mask_ove_all]
            mask_ove_first = df_all_years[outcome + '_observe/expected_First'] == 0
            df_hosp_id_all_years_first = df_all_years[~mask_ove_first]
            mask_ove_reop = df_hosp_id_all_years_reop[outcome + '_observe/expected_Reop'] == 0
            df_hosp_id_all_years_reop_without_0 = df_hosp_id_all_years_reop[~mask_ove_reop]

            outcome_ove_mean_all, outcome_ove_all_25, outcome_ove_all_75, outcome_ove_min_all, outcome_ove_max_all, outcome_ove_std_all = \
                get_stats(df_hosp_id_all_years_ove_all, outcome + '_observe/expected_All')

            outcome_ove_first_median_all, outcome_ove_first_all_25, outcome_ove_first_all_75, outcome_ove_first_min_all, \
            outcome_ove_first_max_all, outcome_ove_first_std_all = get_stats(df_hosp_id_all_years_first,
                                                                             outcome + '_observe/expected_First')

            outcome_ove_reop_mean_all, outcome_ove_reop_all_25, outcome_ove_reop_all_75, outcome_ove_reop_min_all, \
            outcome_ove_reop_max_all, outcome_ove_reop_std_all = get_stats(df_hosp_id_all_years_reop_without_0,
                                                                           outcome + '_observe/expected_Reop')
            experience_of = 'Hospitals'
            if hosp_or_surge == 'surgid':
                experience_of = 'Surgeries'
            if exp.__contains__("Reop"):
                df_all_years['bin_total'] = classify_each_row_to_bin_number(df_all_years, bins_reop, col=exp)
                info_of_bins_reop(outcome, experience_of, exp, type_of_mortality)
            else:
                df_all_years['bin_total'] = classify_each_row_to_bin_number(df_all_years, bins, col=exp)
                info_of_bins(outcome, experience_of, exp, type_of_mortality)
