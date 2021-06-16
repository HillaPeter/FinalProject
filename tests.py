from collections import OrderedDict

import pandas as pd
import matplotlib.pyplot as plt
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


def create_table(df, experince_hosp):
    dict_number_of_hospitals_in_each_bin = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
    dict_number_of_patient_in_each_bin = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
    dict_mortality_in_each_bin = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
    dict_mortality_first_in_each_bin = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
    dict_mortality_reop_in_each_bin = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
    total_hospitals = len(df.drop_duplicates(subset='HospID', keep='first'))

    for index, row in df.iterrows():
        if experince_hosp == 'Reop':
            bin_number = bins_reop(row, experince_hosp)
        else:
            bin_number = bins(row, experince_hosp)
        dict_number_of_patient_in_each_bin[bin_number] += row[experince_hosp]
        dict_number_of_hospitals_in_each_bin[bin_number] += 1
        dict_mortality_in_each_bin[bin_number] += row['Complics_all']
        dict_mortality_reop_in_each_bin[bin_number] += row['Complics_reop']
        dict_mortality_first_in_each_bin[bin_number] += row['Complics_FirstOperation']
    total_patients = sum(dict_number_of_patient_in_each_bin.values())
    return total_hospitals, total_patients, dict_number_of_hospitals_in_each_bin, dict_number_of_patient_in_each_bin, dict_mortality_in_each_bin, dict_mortality_reop_in_each_bin, dict_mortality_first_in_each_bin


def convert_dict_to_df(total_hospitals, total_patients, dict_number_of_hospitals_in_each_bin,
                                dict_number_of_patient_in_each_bin, mort_median, mort_first_median, mort_reop_median,
                                mort_minmax, mort_first_minmax, mort_reop_minmax, mort_std, mort_first_std, mort_reop_std,
                                mort_ove_median, mort_ove_first_median, mort_ove_reop_median,
                                mort_ove_minmax, mort_ove_first_minmax, mort_ove_reop_minmax, mort_ove_std, mort_ove_first_std,
                                mort_ove_reop_std):
    sorted_dict_hospitals = OrderedDict(sorted(dict_number_of_hospitals_in_each_bin.items(), key=lambda t: t[0]))
    sorted_dict_patients = OrderedDict(sorted(dict_number_of_patient_in_each_bin.items(), key=lambda t: t[0]))

    hospitals = [value for (key, value) in sorted_dict_hospitals.items()]
    hospitals.insert(0, total_hospitals)
    patients = [value for (key, value) in sorted_dict_patients.items()]
    patients.insert(0, total_patients)

    all_bins = [hospitals, patients, mort_median, mort_minmax, mort_std, mort_first_median, mort_first_minmax, mort_first_std,
                mort_reop_median, mort_reop_minmax, mort_reop_std, mort_ove_median, mort_ove_minmax, mort_ove_std, mort_ove_first_median, mort_ove_first_minmax, mort_ove_first_std,
                mort_ove_reop_median, mort_ove_reop_minmax, mort_ove_reop_std]

    # Create the pandas DataFrame
    df_final = pd.DataFrame(all_bins,
                            columns=['All', '<100', '100-149', '150-199', '200-299', '300-449', '>=450'])
    df_final.index = ['Hospitals', 'Patients', 'Mortality', 'Mortality [25%,75%]', 'Mortality Std', 'Mortality First',
                      'Mortality First [25%,75%]', 'Mortality First Std', 'Mortality Reop', 'Mortality Reop [25%,75%]', 'Mortality Reop Std',
                      'Mortality OBS/EXP', 'Mortality OBS/EXP [25%,75%]', 'Mortality OBS/EXP Std', 'Mortality OBS/EXP First', 'Mortality OBS/EXP First [25%,75%]', 'Mortality OBS/EXP First Std',
                      'Mortality OBS/EXP Reop', 'Mortality OBS/EXP Reop [25%,75%]', 'Mortality OBS/EXP Reop Std']
    df_final.to_csv('Hospital Volume Category Reop Complics.csv')
    return df_final


'''df - is a specific df for the boxplot
type_id - string of "HospID" or "surgid"
type_outcome - string of "mortalty" or "Complics" '''


def reop_boxplot_three_bins(df, type_id, type_outcome, is_obs_vs_exp=False):
    dict_result = {}
    if is_obs_vs_exp:
        mask = df['count_Reop'] == 0
        column = "log_Reoperation"
    else:
        mask = df['Year_sum_reop'] == 0
        column = f'{type_outcome}_reop_rate'
    df_reop = df[~mask]
    # total_year_sum
    new_df = pd.DataFrame(data=df_reop, columns=[column, 'total_year_avg'])

    new_df['bins'] = pd.qcut(new_df['total_year_avg'], 3, labels=['low', 'mid', 'high'])
    # print(new_df)
    new_df.to_csv(f'box_{type_id}_mort.csv')
    mask = new_df['bins'] == 'low'
    df_low = new_df[mask]
    mask = new_df['bins'] == 'mid'
    df_mid = new_df[mask]
    mask = new_df['bins'] == 'high'
    df_high = new_df[mask]
    data = [df_low[column], df_mid[column], df_high[column]]

    describe_low = dict(df_low[column].describe())
    describe_mid = dict(df_mid[column].describe())
    describe_high = dict(df_high[column].describe())

    # describe_low["median low"] = df_low[column].median()
    # describe_mid["median mid"] = df_mid[column].median()
    # describe_high["median high"] = df_high[column].median()

    range_low = f"{df_low['total_year_avg'].min(): 0.2f}-{df_low['total_year_avg'].max(): 0.2f}"
    range_mid = f"{df_mid['total_year_avg'].min(): 0.2f}-{df_mid['total_year_avg'].max(): 0.2f}"
    range_high = f"{df_high['total_year_avg'].min(): 0.2f}-{df_high['total_year_avg'].max(): 0.2f}"

    describe_low["range bin1"] = range_low
    describe_mid["range bin2"] = range_mid
    describe_high["range bin3"] = range_high

    all_bins = [[value for value in describe_low.values()], [value for value in describe_mid.values()],
                [value for value in describe_high.values()]]

    # Create the pandas DataFrame
    df_final = pd.DataFrame(all_bins,
                            columns=['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max', 'range'])
    df_final = df_final.drop(['min', 'max'], 1)
    df_final.to_csv(f'box_plot_data {type_id}_{type_outcome} obs vs exp - {is_obs_vs_exp}.csv')
    # dict_result["low"] = describe_low
    # dict_result["mid"] = describe_mid
    # dict_result["high"] = describe_high

    r, p = scipy.stats.kruskal(df_low[column], df_mid[column], df_high[column])
    kruskal1 = f" kruskal:  p-value : {p:0.6f}"
    print(kruskal1)



    text = f"low\n ${df_low['total_year_avg'].min(): 0.2f} - ${df_low['total_year_avg'].max(): 0.2f}\n   Mean = ${df_low[column].mean():0.6f} $"
    text2 = f"mid\n ${df_mid['total_year_avg'].min(): 0.2f} - ${df_mid['total_year_avg'].max(): 0.2f}\n   Mean = ${df_mid[column].mean():0.6f} $"
    text3 = f"high\n${df_high['total_year_avg'].min(): 0.2f} - ${df_high['total_year_avg'].max(): 0.2f}\n Mean = ${df_high[column].mean():0.6f} $"
    labels = [text, text2, text3]
    fig1, ax1 = plt.subplots()
    if type_outcome == 'mortalty':
        title = "Mortality"
    else:
        title = "Complication"
    if is_obs_vs_exp:
        full_title = f'{title} {type_id} reop observe vs expected'
    else:
        full_title = f'{title} {type_id} reop'

    ax1.set_title(f"{full_title} boxplot")

    bp = ax1.boxplot(data, patch_artist=True, labels=labels)
    colors = ['pink', 'lightblue', 'palegreen']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    f = lambda c: plt.plot([], color=c, ls="", marker="o")[0]
    plt.legend(handles=[f("pink"), f("lightblue"), f("palegreen")],
               labels=['low', 'mid', 'high'])
    plt.ylabel(f"{full_title} rate")
    plt.show()
    return dict_result


def reop_boxplot_ten_bins(df, type_id, type_outcome, is_obs_vs_exp=False):
    dict_result = {}
    if is_obs_vs_exp:
        mask = df['count_Reop'] == 0
        column = "log_Reoperation"
    else:
        mask = df['Year_sum_reop'] == 0
        column = f'{type_outcome}_reop_rate'
    df_reop = df[~mask]
    # total_year_sum
    new_df = pd.DataFrame(data=df_reop, columns=[column, 'total_year_avg'])

    new_df['bins'] = pd.qcut(new_df['total_year_avg'], 10,
                             labels=['bin1', 'bin2', 'bin3', 'bin4', 'bin5', 'bin6', 'bin7', 'bin8', 'bin9', 'bin10'])
    # print(new_df)
    new_df.to_csv(f'box_{type_id}_{type_outcome}.csv')
    mask = new_df['bins'] == 'bin1'
    df_bin1 = new_df[mask]
    mask = new_df['bins'] == 'bin2'
    df_bin2 = new_df[mask]
    mask = new_df['bins'] == 'bin3'
    df_bin3 = new_df[mask]
    mask = new_df['bins'] == 'bin4'
    df_bin4 = new_df[mask]
    mask = new_df['bins'] == 'bin5'
    df_bin5 = new_df[mask]
    mask = new_df['bins'] == 'bin6'
    df_bin6 = new_df[mask]
    mask = new_df['bins'] == 'bin7'
    df_bin7 = new_df[mask]
    mask = new_df['bins'] == 'bin8'
    df_bin8 = new_df[mask]
    mask = new_df['bins'] == 'bin9'
    df_bin9 = new_df[mask]
    mask = new_df['bins'] == 'bin10'
    df_bin10 = new_df[mask]
    data = [df_bin1[column], df_bin2[column], df_bin3[column], df_bin4[column], df_bin5[column], df_bin6[column],
            df_bin7[column], df_bin8[column], df_bin9[column], df_bin10[column]]

    describe_bin1 = dict(df_bin1[column].describe())
    describe_bin2 = dict(df_bin2[column].describe())
    describe_bin3 = dict(df_bin3[column].describe())
    describe_bin4 = dict(df_bin4[column].describe())
    describe_bin5 = dict(df_bin5[column].describe())
    describe_bin6 = dict(df_bin6[column].describe())
    describe_bin7 = dict(df_bin7[column].describe())
    describe_bin8 = dict(df_bin8[column].describe())
    describe_bin9 = dict(df_bin9[column].describe())
    describe_bin10 = dict(df_bin10[column].describe())

    text1 = f"{df_bin1['total_year_avg'].min(): 0.2f}-{df_bin1['total_year_avg'].max(): 0.2f}"
    text2 = f"{df_bin2['total_year_avg'].min(): 0.2f}-{df_bin2['total_year_avg'].max(): 0.2f}"
    text3 = f"{df_bin3['total_year_avg'].min(): 0.2f}-{df_bin3['total_year_avg'].max(): 0.2f}"
    text4 = f"{df_bin4['total_year_avg'].min(): 0.2f}-{df_bin4['total_year_avg'].max(): 0.2f}"
    text5 = f"{df_bin5['total_year_avg'].min(): 0.2f}-{df_bin5['total_year_avg'].max(): 0.2f}"
    text6 = f"{df_bin6['total_year_avg'].min(): 0.2f}-{df_bin6['total_year_avg'].max(): 0.2f}"
    text7 = f"{df_bin7['total_year_avg'].min(): 0.2f}-{df_bin7['total_year_avg'].max(): 0.2f}"
    text8 = f"{df_bin8['total_year_avg'].min(): 0.2f}-{df_bin8['total_year_avg'].max(): 0.2f}"
    text9 = f"{df_bin9['total_year_avg'].min(): 0.2f}-{df_bin9['total_year_avg'].max(): 0.2f}"
    text10 = f"{df_bin10['total_year_avg'].min(): 0.2f}-{df_bin10['total_year_avg'].max(): 0.2f}"

    describe_bin1["range bin1"] = text1
    describe_bin2["range bin2"] = text2
    describe_bin3["range bin3"] = text3
    describe_bin5["range bin5"] = text4
    describe_bin4["range bin4"] = text5
    describe_bin6["range bin6"] = text6
    describe_bin7["range bin7"] = text7
    describe_bin8["range bin8"] = text8
    describe_bin9["range bin9"] = text9
    describe_bin10["range bin10"] = text10

    all_bins = [[value for value in describe_bin1.values()], [value for value in describe_bin2.values()],
                [value for value in describe_bin3.values()], [value for value in describe_bin4.values()],
                [value for value in describe_bin5.values()], [value for value in describe_bin6.values()],
                [value for value in describe_bin7.values()], [value for value in describe_bin8.values()],
                [value for value in describe_bin9.values()], [value for value in describe_bin10.values()]]

    # Create the pandas DataFrame
    df_final = pd.DataFrame(all_bins, columns=['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max', 'range'])
    df_final = df_final.drop(['min', 'max'], 1)
    df_final.to_csv(f'box_plot_data {type_id}_{type_outcome} obs vs exp - {is_obs_vs_exp}.csv')

    statistic, pvalue = scipy.stats.ttest_ind(df_bin1[column], df_bin10[column])
    t_test = f" t-test:  p-value : {pvalue:0.6f}"
    print(f'box_plot_data {type_id}_{type_outcome} obs vs exp - {is_obs_vs_exp}')
    print(t_test)
    # dict_result["bin1"] = pd.DataFrame([describe_bin1])
    # dict_result["bin2"] = pd.DataFrame([describe_bin2])
    # dict_result["bin3"] = pd.DataFrame([describe_bin3])
    # dict_result["bin4"] = pd.DataFrame([describe_bin4])
    # dict_result["bin5"] = pd.DataFrame([describe_bin5])
    # dict_result["bin6"] = pd.DataFrame([describe_bin6])
    # dict_result["bin7"] = pd.DataFrame([describe_bin7])
    # dict_result["bin8"] = pd.DataFrame([describe_bin8])
    # dict_result["bin9"] = pd.DataFrame([describe_bin9])
    # dict_result["bin10"] = pd.DataFrame([describe_bin10])
    # print(df_final)
    print()

    # dict_result["bin1"] = describe_bin1
    # dict_result["bin2"] = describe_bin2
    # dict_result["bin3"] = describe_bin3
    # dict_result["bin4"] = describe_bin4
    # dict_result["bin5"] = describe_bin5
    # dict_result["bin6"] = describe_bin6
    # dict_result["bin7"] = describe_bin7
    # dict_result["bin8"] = describe_bin8
    # dict_result["bin9"] = describe_bin9
    # dict_result["bin10"] = describe_bin10

    # text = f"low\n ${df_bin1['total_year_avg'].min(): 0.2f} - ${df_bin1['total_year_avg'].max(): 0.2f}\n   Mean = ${df_bin1[column].mean():0.6f} $"
    # text2 = f"mid\n ${df_bin2['total_year_avg'].min(): 0.2f} - ${df_bin2['total_year_avg'].max(): 0.2f}\n   Mean = ${df_bin2[column].mean():0.6f} $"
    # text3 = f"high\n${df_bin3['total_year_avg'].min(): 0.2f} - ${df_bin3['total_year_avg'].max(): 0.2f}\n Mean = ${df_bin3[column].mean():0.6f} $"
    fig1, ax1 = plt.subplots()
    if type_outcome == 'mortalty':
        title = "Mortality"
    else:
        title = "Complication"
    if is_obs_vs_exp:
        full_title = f'{title} {type_id} reop observe vs expected'
    else:
        full_title = f'{title} {type_id} reop'

    ax1.set_title(f"{full_title} boxplot")

    bp = ax1.boxplot(data, patch_artist=True)
    cm = plt.cm.get_cmap('rainbow')
    colors = [cm(val / 10) for val in range(10)]
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    f = lambda c: plt.plot([], color=c, ls="", marker="o")[0]
    # plt.legend(labels=['bin1', 'bin2', 'bin3', 'bin4', 'bin5', 'bin6', 'bin7', 'bin8', 'bin9', 'bin10'])
    plt.ylabel(f"{full_title} rate")
    plt.show()
    return dict_result


def print_dict_result(dict_results, type_id, type_outcome, is_obs_vs_exp=False):
    obs_vs_exp = ""
    if is_obs_vs_exp:
        obs_vs_exp = "observe vs expected"
    print(f"BOX PLOT - {type_outcome} reop of {type_id} {obs_vs_exp}")
    for key in dict_results:
        print(f"{key} : {dict_results[key]}")


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
    r, p = scipy.stats.kruskal(df1[column], df2[column], df3[column], df4[column], df5[column], df6[column])
    print(f'p-value of {column} is : {p}')
    return p


def statistic_test_reop(df1, df2, df3, df4, column):
    r, p = scipy.stats.kruskal(df1[column], df2[column], df3[column], df4[column])
    print(f'p-value of {column} is : {p}')
    return p

def add_item_to_the_begin_of_list(first_item, list_items):
    whole_list = [first_item] + list_items
    return whole_list


def str_to_int(row, col):
    row[col] = (row[col]).replace(',', '')
    return int(float(row[col]))


if __name__ == "__main__":
    #    print("read data and split to bins")

    # print('box plots')
    # df_surge_id = pd.read_csv("total_avg_surgid.csv")
    # df_hosp_id = pd.read_csv("total_avg_hosp_id.csv")
    # df_mort_HospID = pd.read_csv("mortality HospID obs vs expected.csv")

    df_hosp_id_all_years = pd.read_csv('/tmp/pycharm_project_957/hospid_allyears_expec_hospid_three_experiences.csv')
    #df_hosp_id_all_years['total'] = df_hosp_id_all_years.apply(str_to_int, col='total', axis=1)

    df_hosp_id_all_years['bin_total'] = df_hosp_id_all_years.apply(bins_reop, col='Reop', axis=1)
    # df_Hosp['bin_Reop'] = df_Hosp.apply(bins,col='total_year_avg', axis=1)
    # df_hosp_id_all_years['bin_First'] = df_hosp_id_all_years.apply(bins, col='FirstOperation', axis=1)
    # df_total_surgeries = pd.read_csv("new_hosp_total11.csv")
    # observe vs expected
    # df_mort_surgid_obs_vs_exp = pd.read_csv("mortality surgid obs vs expected.csv")
    # df_mort_HospID_obs_vs_exp = pd.read_csv("mortality HospID obs vs expected.csv")
    # df_complics_surgid_obs_vs_exp = pd.read_csv("Complics surgid obs vs expected.csv")
    # df_complics_HospID_obs_vs_exp = pd.read_csv("Complics HospID obs vs expected.csv")

    # split to bins according to CABG
    total_hospitals, total_patients, dict_number_of_hospitals_in_each_bin, dict_number_of_patient_in_each_bin, \
    dict_mortality_in_each_bin, dict_mortality_reop_in_each_bin, dict_mortality_first_in_each_bin = create_table(
        df_hosp_id_all_years, 'Reop')

    # split to bins according to total surgeries
    # total_hospitals, total_patients, dict_number_of_hospitals_in_each_bin, dict_number_of_patient_in_each_bin, \
    # dict_mortality_in_each_bin, dict_mortality_reop_in_each_bin, dict_mortality_first_in_each_bin = create_table(
    #     df_hosp_id_all_years, 'total_surgery_count')

    # split bins according to reop
    # total_hospitals, total_patients, dict_number_of_hospitals_in_each_bin, dict_number_of_patient_in_each_bin, \
    # dict_mortality_in_each_bin, dict_mortality_reop_in_each_bin, dict_mortality_first_in_each_bin = create_table(
    #     df_hosp_id_all_years, 'Reop')


    # ALL
    mortality_median_all, mortality_all_25, mortality_all_75, mortality_min_all, mortality_max_all, mortality_std_all = \
        get_stats(df_hosp_id_all_years, 'Complics_rate_All')

    mortality_first_median_all, mortality_first_all_25, mortality_first_all_75, mortality_first_min_all, \
    mortality_first_max_all, mortality_first_std_all = get_stats(df_hosp_id_all_years, 'Complics_First_rate')

    mask_hosp = df_hosp_id_all_years['Reop'] == 0
    df_hosp_id_all_years_reop = df_hosp_id_all_years[~mask_hosp]

    mortality_reop_median_all, mortality_reop_all_25, mortality_reop_all_75, mortality_reop_min_all, \
    mortality_reop_max_all, mortality_reop_std_all = get_stats(df_hosp_id_all_years_reop, 'Complics_Reop_rate')

    # Obs vs exp
    mask_ove_all = df_hosp_id_all_years['Comp_observe/expected_All'] == 0
    df_hosp_id_all_years_ove_all = df_hosp_id_all_years[~mask_ove_all]
    mask_ove_first = df_hosp_id_all_years['Comp_observe/expected_First'] == 0
    df_hosp_id_all_years_first = df_hosp_id_all_years[~mask_ove_first]
    mask_ove_reop = df_hosp_id_all_years_reop['Comp_observe/expected_Reop'] == 0
    df_hosp_id_all_years_reop_without_0 = df_hosp_id_all_years_reop[~mask_ove_reop]

    mortality_ove_median_all, mortality_ove_all_25, mortality_ove_all_75, mortality_ove_min_all, mortality_ove_max_all, mortality_ove_std_all = \
        get_stats(df_hosp_id_all_years_ove_all, 'Comp_observe/expected_All')

    mortality_ove_first_median_all, mortality_ove_first_all_25, mortality_ove_first_all_75, mortality_ove_first_min_all, \
    mortality_ove_first_max_all, mortality_ove_first_std_all = get_stats(df_hosp_id_all_years_first, 'Comp_observe/expected_First')

    mortality_ove_reop_median_all, mortality_ove_reop_all_25, mortality_ove_reop_all_75, mortality_ove_reop_min_all, \
    mortality_ove_reop_max_all, mortality_ove_reop_std_all = get_stats(df_hosp_id_all_years_reop_without_0, 'Comp_observe/expected_Reop')

    mask = df_hosp_id_all_years['bin_total'] == 1
    df1 = df_hosp_id_all_years[mask]
    mask = df_hosp_id_all_years['bin_total'] == 2
    df2 = df_hosp_id_all_years[mask]
    mask = df_hosp_id_all_years['bin_total'] == 3
    df3 = df_hosp_id_all_years[mask]
    mask = df_hosp_id_all_years['bin_total'] == 4
    df4 = df_hosp_id_all_years[mask]
    # mask = df_hosp_id_all_years['bin_total'] == 5
    # df5 = df_hosp_id_all_years[mask]
    # mask = df_hosp_id_all_years['bin_total'] == 6
    # df6 = df_hosp_id_all_years[mask]

    median_bin1, bin1_25, bin1_75, min_bin1, max_bin1, std_bin1 = get_stats(df1, 'Complics_rate_All')
    median_bin2, bin2_25, bin2_75, min_bin2, max_bin2, std_bin2 = get_stats(df2, 'Complics_rate_All')
    median_bin3, bin3_25, bin3_75, min_bin3, max_bin3, std_bin3 = get_stats(df3, 'Complics_rate_All')
    median_bin4, bin4_25, bin4_75, min_bin4, max_bin4, std_bin4 = get_stats(df4, 'Complics_rate_All')
    # median_bin5, bin5_25, bin5_75, min_bin5, max_bin5, std_bin5 = get_stats(df5, 'Complics_rate_All')
    # median_bin6, bin6_25, bin6_75, min_bin6, max_bin6, std_bin6 = get_stats(df6, 'Complics_rate_All')

    # mortality_median = [median_bin1, median_bin2, median_bin3, median_bin4, median_bin5, median_bin6]
    # mortality_min_max = [(bin1_25, bin1_75), (bin2_25, bin2_75), (bin3_25, bin3_75),
    #                      (bin4_25, bin4_75), (bin5_25, bin5_75), (bin6_25, bin6_75)]
    #
    # mortality_std = [std_bin1, std_bin2, std_bin3, std_bin4, std_bin5, std_bin6]

    mortality_median = [median_bin1, median_bin2, median_bin3, median_bin4]
    mortality_min_max = [(bin1_25, bin1_75), (bin2_25, bin2_75), (bin3_25, bin3_75),
                         (bin4_25, bin4_75)]

    mortality_std = [std_bin1, std_bin2, std_bin3, std_bin4]


    # FIRST
    median_first_bin1, bin1_first_25, bin1_first_75, min_first_bin1, max_first_bin1, std_first_bin1 = get_stats(df1,
                                                                                                'Complics_First_rate')
    median_first_bin2, bin2_first_25, bin2_first_75, min_first_bin2, max_first_bin2, std_first_bin2 = get_stats(df2,
                                                                                                'Complics_First_rate')
    median_first_bin3, bin3_first_25, bin3_first_75, min_first_bin3, max_first_bin3, std_first_bin3 = get_stats(df3,
                                                                                                'Complics_First_rate')
    median_first_bin4, bin4_first_25, bin4_first_75, min_first_bin4, max_first_bin4, std_first_bin4 = get_stats(df4,
                                                                                                'Complics_First_rate')
    # median_first_bin5, bin5_first_25, bin5_first_75, min_first_bin5, max_first_bin5, std_first_bin5 = get_stats(df5,
    #                                                                                             'Complics_First_rate')
    # median_first_bin6, bin6_first_25, bin6_first_75, min_first_bin6, max_first_bin6, std_first_bin6 = get_stats(df6,
    #                                                                                             'Complics_First_rate')

    # mortality_first_median = [median_first_bin1, median_first_bin2, median_first_bin3, median_first_bin4,
    #                           median_first_bin5,
    #                           median_first_bin6]
    # mortality_first_min_max = [(bin1_first_25, bin1_first_75), (bin2_first_25, bin2_first_75),
    #                            (bin3_first_25, bin3_first_75),
    #                            (bin4_first_25, bin4_first_75), (bin5_first_25, bin5_first_75),
    #                            (bin6_first_25, bin6_first_75)]

    # mortality_first_std = [std_first_bin1, std_first_bin2, std_first_bin3, std_first_bin4, std_first_bin5, std_first_bin6]

    mortality_first_median = [median_first_bin1, median_first_bin2, median_first_bin3, median_first_bin4]
    mortality_first_min_max = [(bin1_first_25, bin1_first_75), (bin2_first_25, bin2_first_75),
                               (bin3_first_25, bin3_first_75),
                               (bin4_first_25, bin4_first_75)]

    mortality_first_std = [std_first_bin1, std_first_bin2, std_first_bin3, std_first_bin4]



    # RE-OP
    mask_hosp = df_hosp_id_all_years['Reop'] == 0
    df_hosp_id_all_years_reop = df_hosp_id_all_years[~mask_hosp]

    mask = df1['Reop'] == 0
    df1_reop = df1[~mask]
    mask = df2['Reop'] == 0
    df2_reop = df2[~mask]
    mask = df3['Reop'] == 0
    df3_reop = df3[~mask]
    mask = df4['Reop'] == 0
    df4_reop = df4[~mask]
    # mask = df5['Reop'] == 0
    # df5_reop = df5[~mask]
    # mask = df6['Reop'] == 0
    # df6_reop = df6[~mask]

    median_reop_bin1, bin1_reop_25, bin1_reop_75, min_reop_bin1, max_reop_bin1, std_reop_bin1 = get_stats(df1_reop, 'Complics_Reop_rate')
    median_reop_bin2, bin2_reop_25, bin2_reop_75, min_reop_bin2, max_reop_bin2, std_reop_bin2 = get_stats(df2_reop, 'Complics_Reop_rate')
    median_reop_bin3, bin3_reop_25, bin3_reop_75, min_reop_bin3, max_reop_bin3, std_reop_bin3 = get_stats(df3_reop, 'Complics_Reop_rate')
    median_reop_bin4, bin4_reop_25, bin4_reop_75, min_reop_bin4, max_reop_bin4, std_reop_bin4 = get_stats(df4_reop, 'Complics_Reop_rate')
    # median_reop_bin5, bin5_reop_25, bin5_reop_75, min_reop_bin5, max_reop_bin5, std_reop_bin5 = get_stats(df5_reop, 'Complics_Reop_rate')
    # median_reop_bin6, bin6_reop_25, bin6_reop_75, min_reop_bin6, max_reop_bin6, std_reop_bin6 = get_stats(df6_reop, 'Complics_Reop_rate')

    # mortality_reop_median = [median_reop_bin1, median_reop_bin2, median_reop_bin3, median_reop_bin4, median_reop_bin5,
    #                          median_reop_bin6]
    # mortality_reop_min_max = [(bin1_reop_25, bin1_reop_75), (bin2_reop_25, bin2_reop_75), (bin3_reop_25, bin3_reop_75),
    #                           (bin4_reop_25, bin4_reop_75), (bin5_reop_25, bin5_reop_75), (bin6_reop_25, bin6_reop_75)]
    #
    # mortality_reop_std = [std_reop_bin1, std_reop_bin2, std_reop_bin3, std_reop_bin4, std_reop_bin5, std_reop_bin6]


    mortality_reop_median = [median_reop_bin1, median_reop_bin2, median_reop_bin3, median_reop_bin4]
    mortality_reop_min_max = [(bin1_reop_25, bin1_reop_75), (bin2_reop_25, bin2_reop_75), (bin3_reop_25, bin3_reop_75),
                              (bin4_reop_25, bin4_reop_75)]

    mortality_reop_std = [std_reop_bin1, std_reop_bin2, std_reop_bin3, std_reop_bin4]




    # OBS/EXP
    median_ove_bin1, bin1_ove_25, bin1_ove_75, min_ove_bin1, max_ove_bin1, std_ove_bin1 = get_stats(df1, 'Comp_observe/expected_All')
    median_ove_bin2, bin2_ove_25, bin2_ove_75, min_ove_bin2, max_ove_bin2, std_ove_bin2 = get_stats(df2, 'Comp_observe/expected_All')
    median_ove_bin3, bin3_ove_25, bin3_ove_75, min_ove_bin3, max_ove_bin3, std_ove_bin3 = get_stats(df3, 'Comp_observe/expected_All')
    median_ove_bin4, bin4_ove_25, bin4_ove_75, min_ove_bin4, max_ove_bin4, std_ove_bin4 = get_stats(df4, 'Comp_observe/expected_All')
    # median_ove_bin5, bin5_ove_25, bin5_ove_75, min_ove_bin5, max_ove_bin5, std_ove_bin5 = get_stats(df5, 'Comp_observe/expected_All')
    # median_ove_bin6, bin6_ove_25, bin6_ove_75, min_ove_bin6, max_ove_bin6, std_ove_bin6 = get_stats(df6, 'Comp_observe/expected_All')

    # mortality_ove_median = [median_ove_bin1, median_ove_bin2, median_ove_bin3, median_ove_bin4, median_ove_bin5, median_ove_bin6]
    # mortality_ove_min_max = [(bin1_ove_25, bin1_ove_75), (bin2_ove_25, bin2_ove_75), (bin3_ove_25, bin3_ove_75),
    #                      (bin4_ove_25, bin4_ove_75), (bin5_ove_25, bin5_ove_75), (bin6_ove_25, bin6_ove_75)]
    #
    # mortality_ove_std = [std_ove_bin1, std_ove_bin2, std_ove_bin3, std_ove_bin4, std_ove_bin5, std_ove_bin6]

    mortality_ove_median = [median_ove_bin1, median_ove_bin2, median_ove_bin3, median_ove_bin4]
    mortality_ove_min_max = [(bin1_ove_25, bin1_ove_75), (bin2_ove_25, bin2_ove_75), (bin3_ove_25, bin3_ove_75),
                             (bin4_ove_25, bin4_ove_75)]

    mortality_ove_std = [std_ove_bin1, std_ove_bin2, std_ove_bin3, std_ove_bin4]


    # OBS/EXP First

    mask = df1['Comp_observe/expected_First'] == 0
    df1_first_ove = df1[~mask]
    mask = df2['Comp_observe/expected_First'] == 0
    df2_first_ove = df2[~mask]
    mask = df3['Comp_observe/expected_First'] == 0
    df3_first_ove = df3[~mask]
    mask = df4['Comp_observe/expected_First'] == 0
    df4_first_ove = df4[~mask]
    # mask = df5['Comp_observe/expected_First'] == 0
    # df5_first_ove = df5[~mask]
    # mask = df6['Comp_observe/expected_First'] == 0
    # df6_first_ove = df6[~mask]


    median_ove_first_bin1, bin1_ove_first_25, bin1_ove_first_75, min_ove_first_bin1, max_ove_first_bin1, std_ove_first_bin1 = get_stats(df1_first_ove,
                                                                                                                    'Comp_observe/expected_First')
    median_ove_first_bin2, bin2_ove_first_25, bin2_ove_first_75, min_ove_first_bin2, max_ove_first_bin2,std_ove_first_bin2 = get_stats(df2_first_ove,
                                                                                                                    'Comp_observe/expected_First')
    median_ove_first_bin3, bin3_ove_first_25, bin3_ove_first_75, min_ove_first_bin3, max_ove_first_bin3, std_ove_first_bin3 = get_stats(df3_first_ove,
                                                                                                                    'Comp_observe/expected_First')
    median_ove_first_bin4, bin4_ove_first_25, bin4_ove_first_75, min_ove_first_bin4, max_ove_first_bin4, std_ove_first_bin4 = get_stats(df4_first_ove,
                                                                                                                    'Comp_observe/expected_First')
    # median_ove_first_bin5, bin5_ove_first_25, bin5_ove_first_75, min_ove_first_bin5, max_ove_first_bin5, std_ove_first_bin5 = get_stats(df5_first_ove,
    #                                                                                                                 'Comp_observe/expected_First')
    # median_ove_first_bin6, bin6_ove_first_25, bin6_ove_first_75, min_ove_first_bin6, max_ove_first_bin6, std_ove_first_bin6 = get_stats(df6_first_ove,
    #                                                                                                                 'Comp_observe/expected_First')

    # mortality_ove_first_median = [median_ove_first_bin1, median_ove_first_bin2, median_ove_first_bin3,
    #                               median_ove_first_bin4,
    #                               median_ove_first_bin5, median_ove_first_bin6]
    # mortality_ove_first_min_max = [(bin1_ove_first_25, bin1_ove_first_75), (bin2_ove_first_25, bin2_ove_first_75),
    #                                (bin3_ove_first_25, bin3_ove_first_75),
    #                                (bin4_ove_first_25, bin4_ove_first_75), (bin5_ove_first_25, bin5_ove_first_75),
    #                                (bin6_ove_first_25, bin6_ove_first_75)]
    #
    # mortality_ove_first_std = [std_ove_first_bin1, std_ove_first_bin2, std_ove_first_bin3, std_ove_first_bin4, std_ove_first_bin5, std_ove_first_bin6]

    mortality_ove_first_median = [median_ove_first_bin1, median_ove_first_bin2, median_ove_first_bin3,
                                  median_ove_first_bin4]
    mortality_ove_first_min_max = [(bin1_ove_first_25, bin1_ove_first_75), (bin2_ove_first_25, bin2_ove_first_75),
                                   (bin3_ove_first_25, bin3_ove_first_75),
                                   (bin4_ove_first_25, bin4_ove_first_75)]

    mortality_ove_first_std = [std_ove_first_bin1, std_ove_first_bin2, std_ove_first_bin3, std_ove_first_bin4]


    # OBS/EXP RE-OP
    mask = df1_reop['Comp_observe/expected_Reop'] == 0
    df1_reop_ove = df1_reop[~mask]
    mask = df2_reop['Comp_observe/expected_Reop'] == 0
    df2_reop_ove = df2_reop[~mask]
    mask = df3_reop['Comp_observe/expected_Reop'] == 0
    df3_reop_ove = df3_reop[~mask]
    mask = df4_reop['Comp_observe/expected_Reop'] == 0
    df4_reop_ove = df4_reop[~mask]
    # mask = df5_reop['Comp_observe/expected_Reop'] == 0
    # df5_reop_ove = df5_reop[~mask]
    # mask = df6_reop['Comp_observe/expected_Reop'] == 0
    # df6_reop_ove = df6_reop[~mask]


    median_ove_reop_bin1, bin1_ove_reop_25, bin1_ove_reop_75, min_ove_reop_bin1, max_ove_reop_bin1, std_ove_reop_bin1 = get_stats(df1_reop_ove,
                                                                                                               'Comp_observe/expected_Reop')
    median_ove_reop_bin2, bin2_ove_reop_25, bin2_ove_reop_75, min_ove_reop_bin2, max_ove_reop_bin2, std_ove_reop_bin2 = get_stats(df2_reop_ove,
                                                                                                               'Comp_observe/expected_Reop')
    median_ove_reop_bin3, bin3_ove_reop_25, bin3_ove_reop_75, min_ove_reop_bin3, max_ove_reop_bin3, std_ove_reop_bin3 = get_stats(df3_reop_ove,
                                                                                                               'Comp_observe/expected_Reop')
    median_ove_reop_bin4, bin4_ove_reop_25, bin4_ove_reop_75, min_ove_reop_bin4, max_ove_reop_bin4, std_ove_reop_bin4 = get_stats(df4_reop_ove,
                                                                                                               'Comp_observe/expected_Reop')
    # median_ove_reop_bin5, bin5_ove_reop_25, bin5_ove_reop_75, min_ove_reop_bin5, max_ove_reop_bin5, std_ove_reop_bin5 = get_stats(df5_reop_ove,
    #                                                                                                            'Comp_observe/expected_Reop')
    # median_ove_reop_bin6, bin6_ove_reop_25, bin6_ove_reop_75, min_ove_reop_bin6, max_ove_reop_bin6, std_ove_reop_bin6 = get_stats(df6_reop_ove,
    #                                                                                                            'Comp_observe/expected_Reop')

    # mortality_ove_reop_median = [median_ove_reop_bin1, median_ove_reop_bin2, median_ove_reop_bin3,
    #                              median_ove_reop_bin4,
    #                              median_ove_reop_bin5, median_ove_reop_bin6]
    # mortality_ove_reop_min_max = [(bin1_ove_reop_25, bin1_ove_reop_75), (bin2_ove_reop_25, bin2_ove_reop_75),
    #                               (bin3_ove_reop_25, bin3_ove_reop_75),
    #                               (bin4_ove_reop_25, bin4_ove_reop_75), (bin5_ove_reop_25, bin5_ove_reop_75),
    #                               (bin6_ove_reop_25, bin6_ove_reop_75)]
    #
    # mortality_ove_reop_std = [std_ove_reop_bin1, std_ove_reop_bin2, std_ove_reop_bin3, std_ove_reop_bin4, std_ove_reop_bin5, std_ove_reop_bin6]

    mortality_ove_reop_median = [median_ove_reop_bin1, median_ove_reop_bin2, median_ove_reop_bin3,
                                 median_ove_reop_bin4]
    mortality_ove_reop_min_max = [(bin1_ove_reop_25, bin1_ove_reop_75), (bin2_ove_reop_25, bin2_ove_reop_75),
                                  (bin3_ove_reop_25, bin3_ove_reop_75),
                                  (bin4_ove_reop_25, bin4_ove_reop_75)]

    mortality_ove_reop_std = [std_ove_reop_bin1, std_ove_reop_bin2, std_ove_reop_bin3, std_ove_reop_bin4]





    # STATISTIC TEST
    # p_mort_all = statistic_test(df1, df2, df3, df4, df5, df6, 'Complics_rate_All')
    # p_mort_first = statistic_test(df1, df2, df3, df4, df5, df6, 'Complics_First_rate')
    # p_mort_reop = statistic_test(df1_reop, df2_reop, df3_reop, df4_reop, df5_reop, df6_reop, 'Complics_Reop_rate')
    # p_mort_ove = statistic_test(df1, df2, df3, df4, df5, df6, 'Comp_observe/expected_All')
    # p_mort_ove_first = statistic_test(df1, df2, df3, df4, df5, df6, 'Comp_observe/expected_First')
    # p_mort_ove_reop = statistic_test(df1_reop, df2_reop, df3_reop, df4_reop, df5_reop, df6_reop, 'Comp_observe/expected_Reop')

    p_mort_all = statistic_test_reop(df1, df2, df3, df4, 'Complics_rate_All')
    p_mort_first = statistic_test_reop(df1, df2, df3, df4, 'Complics_First_rate')
    p_mort_reop = statistic_test_reop(df1_reop, df2_reop, df3_reop, df4_reop, 'Complics_Reop_rate')
    p_mort_ove = statistic_test_reop(df1, df2, df3, df4, 'Comp_observe/expected_All')
    p_mort_ove_first = statistic_test_reop(df1, df2, df3, df4, 'Comp_observe/expected_First')
    p_mort_ove_reop = statistic_test_reop(df1_reop, df2_reop, df3_reop, df4_reop, 'Comp_observe/expected_Reop')





    mort_median = add_item_to_the_begin_of_list(mortality_median_all, mortality_median)
    mort_first_median = add_item_to_the_begin_of_list(mortality_first_median_all, mortality_first_median)
    mort_reop_median = add_item_to_the_begin_of_list(mortality_reop_median_all, mortality_reop_median)
    mort_minmax = add_item_to_the_begin_of_list((mortality_min_all, mortality_max_all), mortality_min_max)
    mort_first_minmax = add_item_to_the_begin_of_list((mortality_first_min_all, mortality_first_max_all), mortality_first_min_max)
    mort_reop_minmax = add_item_to_the_begin_of_list((mortality_reop_min_all, mortality_reop_max_all), mortality_reop_min_max)
    mort_std = [mortality_std_all] + mortality_std
    mort_first_std = [mortality_first_std_all] + mortality_first_std
    mort_reop_std = [mortality_reop_std_all] + mortality_reop_std
    mort_ove_median = add_item_to_the_begin_of_list(mortality_ove_median_all, mortality_ove_median)  # obs vs exp median
    mort_ove_first_median = add_item_to_the_begin_of_list(mortality_ove_first_median_all, mortality_ove_first_median)
    mort_ove_reop_median = add_item_to_the_begin_of_list(mortality_ove_reop_median_all, mortality_ove_reop_median)
    mort_ove_minmax = add_item_to_the_begin_of_list((mortality_ove_all_25, mortality_ove_all_75), mortality_ove_min_max)  # obs vs exp 25-75
    mort_ove_first_minmax = add_item_to_the_begin_of_list((mortality_ove_first_all_25, mortality_ove_first_all_75), mortality_ove_first_min_max)
    mort_ove_reop_minmax = add_item_to_the_begin_of_list((mortality_ove_reop_all_25, mortality_ove_reop_all_75), mortality_ove_reop_min_max)
    mort_ove_std = add_item_to_the_begin_of_list(mortality_ove_std_all, mortality_ove_std)  # obs vs exp std
    mort_ove_first_std = add_item_to_the_begin_of_list(mortality_ove_first_std_all, mortality_ove_first_std)
    mort_ove_reop_std = add_item_to_the_begin_of_list(mortality_ove_reop_std_all, mortality_ove_reop_std)


    # COMBINE ALL TOGETHER
    df_All = convert_dict_to_df(total_hospitals, total_patients, dict_number_of_hospitals_in_each_bin,
                                dict_number_of_patient_in_each_bin, mort_median, mort_first_median, mort_reop_median,
                                mort_minmax, mort_first_minmax, mort_reop_minmax, mort_std, mort_first_std, mort_reop_std,
                                mort_ove_median, mort_ove_first_median, mort_ove_reop_median,
                                mort_ove_minmax, mort_ove_first_minmax, mort_ove_reop_minmax, mort_ove_std, mort_ove_first_std,
                                mort_ove_reop_std)


    # dict_mortality_surge_id = reop_boxplot_three_bins(df_surge_id, "surgid", "mortalty")
    # print_dict_result(dict_mortality_surge_id, "surgeid", "mortality")
    # dict_mortality_site_id = reop_boxplot_three_bins(df_hosp_id, "HospID", "mortalty")
    # print_dict_result(dict_mortality_site_id, "HospID", "mortality")
    # dict_complics_surge_id = reop_boxplot_three_bins(df_surge_id, "surgid", "Complics")
    # print_dict_result(dict_complics_surge_id, "surgeid", "complications")
    # dict_complics_site_id = reop_boxplot_three_bins(df_hosp_id, "HospID", "Complics")
    # print_dict_result(dict_complics_site_id, "HospID", "complications")

    # dict_mort_surge_id_obs_vs_exp = reop_boxplot_three_bins(df_mort_surgid_obs_vs_exp, "surgid", "mortalty", True)
    # print_dict_result(dict_mort_surge_id_obs_vs_exp, "surgeid", "mortality", True)
    # dict_mort_hosp_id_obs_vs_exp = reop_boxplot_three_bins(df_mort_HospID_obs_vs_exp, "HospID", "mortalty", True)
    # print_dict_result(dict_mort_hosp_id_obs_vs_exp, "HospID", "mortality", True)
    # dict_complics_surge_id_obs_vs_exp = reop_boxplot_three_bins(df_complics_surgid_obs_vs_exp, "surgid", "Complics", True)
    # print_dict_result(dict_complics_surge_id_obs_vs_exp, "surgeid", "complications", True)
    # dict_complics_hosp_id_obs_vs_exp = reop_boxplot_three_bins(df_complics_HospID_obs_vs_exp, "HospID", "Complics", True)
    # print_dict_result(dict_complics_hosp_id_obs_vs_exp, "HospID", "complications", True)

    # # ten bins
    # # print("TEN BINS")
    # dict_mortality_surge_id_ten_bins = reop_boxplot_ten_bins(df_surge_id, "surgid", "mortalty")
    # # print_dict_result(dict_mortality_surge_id_ten_bins, "surgeid", "mortality")
    # dict_mortality_hosp_id_ten_bins = reop_boxplot_ten_bins(df_hosp_id, "HospID", "mortalty")
    # # print_dict_result(dict_mortality_hosp_id_ten_bins, "HospID", "mortality")
    # dict_complics_surge_id_ten_bins = reop_boxplot_ten_bins(df_surge_id, "surgid", "Complics")
    # # print_dict_result(dict_complics_surge_id_ten_bins, "surgeid", "complications")
    # dict_complics_hosp_id_ten_bins = reop_boxplot_ten_bins(df_hosp_id, "HospID", "Complics")
    # # print_dict_result(dict_complics_hosp_id_ten_bins, "HospID", "complications")
    #
    # # observe vs expected
    # dict_mort_surge_id_obs_vs_exp_ten_bins = reop_boxplot_ten_bins(df_mort_surgid_obs_vs_exp, "surgid", "mortalty",
    #                                                                True)
    # # print_dict_result(dict_mort_surge_id_obs_vs_exp_ten_bins, "surgeid", "mortality", True)
    # dict_mort_hosp_id_obs_vs_exp_ten_bins = reop_boxplot_ten_bins(df_mort_HospID_obs_vs_exp, "HospID", "mortalty", True)
    # # print_dict_result(dict_mort_hosp_id_obs_vs_exp_ten_bins, "HospID", "mortality", True)
    # dict_complics_surge_id_obs_vs_exp_ten_bins = reop_boxplot_ten_bins(df_complics_surgid_obs_vs_exp, "surgid",
    #                                                                    "Complics", True)
    # # print_dict_result(dict_complics_surge_id_obs_vs_exp_ten_bins, "surgeid", "complications", True)
    # dict_complics_hosp_id_obs_vs_exp_ten_bins = reop_boxplot_ten_bins(df_complics_HospID_obs_vs_exp, "HospID",
    #                                                                   "Complics", True)
    # print_dict_result(dict_complics_hosp_id_obs_vs_exp_ten_bins, "HospID", "complications", True)

    # total surgeries
    # print("total surgeries")
    # dict_mortality_site_id = reop_boxplot_three_bins(df_total_surgeries, "HospID", "mortalty")
