import pandas as pd
import matplotlib.pyplot as plt

# df = pd.DataFrame({'Foo': ['A','B','C','D','E'],
# 'Score1': [4,6,2,7,8]
# })
# df.columns=['Foo','Score1']
# df2 = pd.DataFrame({'Foo': ['A','C','D','E'],
# 'Score2': [5,10,10,5]
# })
# df2.columns=['Foo','Score2']
# result = pd.merge(df, df2, on='Foo', how='left')
# print(result)
# result=result.dropna()
# result["Score1/Score2"] = result["Score1"] / result["Score2"]
# # result = result.iloc[1:]
# print(result)

# BOX PLOTS

'''df - is a specific df for the boxplot
type_id - string of "siteid" or "surgid"
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
    # print("data about 'low' bin\n", df_low.describe())
    # print("data about 'mid' bin\n", df_mid.describe())
    # print("data about 'high' bin\n", df_high.describe())
    describe_low = dict(df_low[column].describe())
    describe_mid = dict(df_mid[column].describe())
    describe_high = dict(df_high[column].describe())
    describe_low["median low"] = df_low[column].median()
    describe_mid["median mid"] = df_mid[column].median()
    describe_high["median high"] = df_high[column].median()
    dict_result["low"] = describe_low
    dict_result["mid"] = describe_mid
    dict_result["high"] = describe_high
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

    new_df['bins'] = pd.qcut(new_df['total_year_avg'], 10, labels=['bin1', 'bin2', 'bin3', 'bin4', 'bin5', 'bin6', 'bin7', 'bin8', 'bin9', 'bin10'])
    # print(new_df)
    new_df.to_csv(f'box_{type_id}_mort.csv')
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
    data = [df_bin1[column], df_bin2[column], df_bin3[column], df_bin4[column], df_bin5[column], df_bin6[column], df_bin7[column], df_bin8[column], df_bin9[column], df_bin10[column]]
    # print("data about 'low' bin\n", df_low.describe())
    # print("data about 'mid' bin\n", df_mid.describe())
    # print("data about 'high' bin\n", df_high.describe())
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

    describe_bin1["median bin1"] = df_bin1[column].median()
    describe_bin2["median bin2"] = df_bin2[column].median()
    describe_bin3["median bin3"] = df_bin3[column].median()
    describe_bin4["median bin4"] = df_bin4[column].median()
    describe_bin5["median bin5"] = df_bin5[column].median()
    describe_bin6["median bin6"] = df_bin6[column].median()
    describe_bin7["median bin7"] = df_bin7[column].median()
    describe_bin8["median bin8"] = df_bin8[column].median()
    describe_bin9["median bin9"] = df_bin9[column].median()
    describe_bin10["median bin10"] = df_bin10[column].median()

    dict_result["bin1"] = describe_bin1
    dict_result["bin2"] = describe_bin2
    dict_result["bin3"] = describe_bin3
    dict_result["bin4"] = describe_bin4
    dict_result["bin5"] = describe_bin5
    dict_result["bin6"] = describe_bin6
    dict_result["bin7"] = describe_bin7
    dict_result["bin8"] = describe_bin8
    dict_result["bin9"] = describe_bin9
    dict_result["bin10"] = describe_bin10

    # text = f"low\n ${df_bin1['total_year_avg'].min(): 0.2f} - ${df_bin1['total_year_avg'].max(): 0.2f}\n   Mean = ${df_bin1[column].mean():0.6f} $"
    # text2 = f"mid\n ${df_bin2['total_year_avg'].min(): 0.2f} - ${df_bin2['total_year_avg'].max(): 0.2f}\n   Mean = ${df_bin2[column].mean():0.6f} $"
    # text3 = f"high\n${df_bin3['total_year_avg'].min(): 0.2f} - ${df_bin3['total_year_avg'].max(): 0.2f}\n Mean = ${df_bin3[column].mean():0.6f} $"
    # labels = [text, text2, text3]
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
    colors = ['pink', 'lightblue', 'palegreen']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    f = lambda c: plt.plot([], color=c, ls="", marker="o")[0]
    plt.legend(labels=['bin1', 'bin2', 'bin3', 'bin4', 'bin5', 'bin6', 'bin7', 'bin8', 'bin9', 'bin10'])
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

if __name__ == "__main__":
    print('box plots')
    df_surge_id = pd.read_csv("total_avg_surgid.csv")
    df_site_id = pd.read_csv("total_avg_site_id.csv")
    # dict_mortality_surge_id = reop_boxplot_three_bins(df_surge_id, "surgid", "mortalty")
    # print_dict_result(dict_mortality_surge_id, "surgeid", "mortality")
    # dict_mortality_site_id = reop_boxplot_three_bins(df_site_id, "siteid", "mortalty")
    # print_dict_result(dict_mortality_site_id, "siteid", "mortality")
    # dict_complics_surge_id = reop_boxplot_three_bins(df_surge_id, "surgid", "Complics")
    # print_dict_result(dict_complics_surge_id, "surgeid", "complications")
    # dict_complics_site_id = reop_boxplot_three_bins(df_site_id, "siteid", "Complics")
    # print_dict_result(dict_complics_site_id, "siteid", "complications")
    #
    # # observe vs expected
    # df_mort_surgid_obs_vs_exp = pd.read_csv("mortality surgid obs vs expected.csv")
    # df_mort_siteid_obs_vs_exp = pd.read_csv("mortality siteid obs vs expected.csv")
    # df_complics_surgid_obs_vs_exp = pd.read_csv("Complics surgid obs vs expected.csv")
    # df_complics_siteid_obs_vs_exp = pd.read_csv("Complics siteid obs vs expected.csv")
    #
    # dict_mort_surge_id_obs_vs_exp = reop_boxplot_three_bins(df_mort_surgid_obs_vs_exp, "surgid", "mortalty", True)
    # print_dict_result(dict_mort_surge_id_obs_vs_exp, "surgeid", "mortality", True)
    # dict_mort_site_id_obs_vs_exp = reop_boxplot_three_bins(df_mort_siteid_obs_vs_exp, "siteid", "mortalty", True)
    # print_dict_result(dict_mort_site_id_obs_vs_exp, "siteid", "mortality", True)
    # dict_complics_surge_id_obs_vs_exp = reop_boxplot_three_bins(df_complics_surgid_obs_vs_exp, "surgid", "Complics", True)
    # print_dict_result(dict_complics_surge_id_obs_vs_exp, "surgeid", "complications", True)
    # dict_complics_site_id_obs_vs_exp = reop_boxplot_three_bins(df_complics_siteid_obs_vs_exp, "siteid", "Complics", True)
    # print_dict_result(dict_complics_site_id_obs_vs_exp, "siteid", "complications", True)

    #ten bins
    print("TEN BINS")
    # dict_mortality_surge_id_ten_bins = reop_boxplot_three_bins(df_surge_id, "surgid", "mortalty")
    # print_dict_result(dict_mortality_surge_id_ten_bins, "surgeid", "mortality")
    # dict_mortality_site_id_ten_bins = reop_boxplot_three_bins(df_site_id, "siteid", "mortalty")
    # print_dict_result(dict_mortality_site_id_ten_bins, "siteid", "mortality")
    # dict_complics_surge_id_ten_bins = reop_boxplot_three_bins(df_surge_id, "surgid", "Complics")
    # print_dict_result(dict_complics_surge_id_ten_bins, "surgeid", "complications")
    # dict_complics_site_id_ten_bins = reop_boxplot_three_bins(df_site_id, "siteid", "Complics")
    # print_dict_result(dict_complics_site_id_ten_bins, "siteid", "complications")

    # observe vs expected
    # dict_mort_surge_id_obs_vs_exp_ten_bins = reop_boxplot_three_bins(df_mort_surgid_obs_vs_exp, "surgid", "mortalty", True)
    # print_dict_result(dict_mort_surge_id_obs_vs_exp_ten_bins, "surgeid", "mortality", True)
    # dict_mort_site_id_obs_vs_exp_ten_bins = reop_boxplot_three_bins(df_mort_siteid_obs_vs_exp, "siteid", "mortalty", True)
    # print_dict_result(dict_mort_site_id_obs_vs_exp_ten_bins, "siteid", "mortality", True)
    # dict_complics_surge_id_obs_vs_exp_ten_bins = reop_boxplot_three_bins(df_complics_surgid_obs_vs_exp, "surgid", "Complics", True)
    # print_dict_result(dict_complics_surge_id_obs_vs_exp_ten_bins, "surgeid", "complications", True)
    # dict_complics_site_id_obs_vs_exp_ten_bins = reop_boxplot_three_bins(df_complics_siteid_obs_vs_exp, "siteid", "Complics", True)
    # print_dict_result(dict_complics_site_id_obs_vs_exp_ten_bins, "siteid", "complications", True)
