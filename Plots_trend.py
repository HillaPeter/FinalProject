import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import r2_score
import scipy.stats

def boxplot_withback(col, title,xt,yt, data,  y):
    box_plot = sns.boxplot(x=col, y=y, data=data, palette="Blues")

    ax = box_plot.axes
    lines = ax.get_lines()
    categories = ax.get_xticks()

    for cat in categories:
        # every 4th line at the interval of 6 is median line
        # 0 -> p25 1 -> p75 2 -> lower whisker 3 -> upper whisker 4 -> p50 5 -> upper extreme value
        y = round(lines[4 + cat * 6].get_ydata()[0], 3)

        ax.text(
            cat,
            y,
            f'{y}',
            ha='center',
            va='center',
            fontweight='bold',
            size=10,
            color='white',
            bbox=dict(facecolor='#445A64'))

    box_plot.figure.tight_layout()
    plt.title(title)
    plt.xlabel(xt)
    plt.ylabel(yt)
    plt.show()


def plot_hosp(col, title,xt,yt, data, x, y):

    fig = plt.gcf()
    colors =sns.color_palette("pastel")
    sns.lmplot(x=x, y=y, hue=col, data=data,palette=colors,scatter_kws={"s": 20}, line_kws={"lw":1,'color': 'red'})
    plt.title(title)
    plt.xlabel(xt)
    plt.ylabel(yt)
    plt.show()
    #plt.savefig(fig, bbox_inches='tight')
    # ax.set_xlabel(xlabel='count of albums per genre', fontsize=16)
    # # ax.set_ylabel(ylabel='Year', fontsize=16)
    # ax.set_title(label='Genre distribution', fontsize=20)
    plt.figure(figsize=(12, 4))
    abx = sns.regplot(x=x, y=y,  data=data,lowess=True,
                     scatter_kws={"color": "gray"}, line_kws={"color": "red"})

    plt.title(title)
    plt.xlabel(xt)
    plt.ylabel(yt)
    plt.show()
   # plt.savefig(abx,bbox_inches='tight')
    # sns.boxplot(x=df_Hosp[col], y=df_Hosp["mortalty_rate"], palette="Blues",width=0.3)
    # plt.show()

    # ax = sns.boxplot(x=data[col], y=data[y])
    #
    # medians = data.groupby([col])[y].median().values
    # median_labels = [str(np.round(s, 2)) for s in medians]
    #
    # pos = range(len(medians))
    # for tick, label in zip(pos, ax.get_xticklabels()):
    #     ax.text(pos[tick], medians[tick] - 2, median_labels[tick],
    #             horizontalalignment='center', size='x-small', color='black', weight='semibold')
    # plt.title(title)
    # plt.xlabel(xt)
    # plt.ylabel(yt)
    # plt.show()

    boxplot_withback(col,title,xt,yt,data,y)

    # plt.figure(figsize=(12, 4))
    # colors = sns.color_palette("pastel")
    # sns.lmplot(x=x, y=y,  data=data, palette="Set2", lowess=True,line_kws={"color": "red"})
    #
    # plt.title(title)
    # plt.xlabel(xt)
    # plt.ylabel(yt)
    # plt.show()


#################################################################################
# mortality-total_cardiac
# #hosp -total surgery count
# #surg'total cardiac surgery'
# df_Hosp = pd.read_csv("hospid_allyears_expec_hospid_STSRCMM_div.csv")
# plot_hosp("bin_total_cardiac", "hospid mortality total rate", "Year total operation", "mortality rate",df_Hosp,'total surgery count','mort_rate_All')
# mask_bin = df_Hosp['Reop']== 0
# bin_reop_df = df_Hosp[~mask_bin]
# mask_bin = df_Hosp['FirstOperation']== 0
# bin_op_df = df_Hosp[~mask_bin]
# plot_hosp("bin_total_cardiac", "hospid mortality First operation rate", "Year total operation", "mortality first rate",bin_op_df,'total surgery count','Mortality_First_rate')
# plot_hosp("bin_total_cardiac", "hospid mortality Reoperation rate", "Year total operation ", "mortality Reoperation rate",bin_reop_df,'total surgery count','Mortality_Reop_rate')
#
# mask_1 = bin_op_df['Mort_observe/expected_First'] == 0
# bin_op_df = bin_op_df[~mask_1]
# mask_2 = bin_reop_df['Mort_observe/expected_Reop'] == 0
# bin_reop_df = bin_reop_df[~mask_2]
# plot_hosp("bin_total_cardiac", "hospid mortality obs vs expected First operation ", "Year total operation", "log obs vs expected first ",bin_op_df,'total surgery count','log_First_Mort')
# plot_hosp("bin_total_cardiac", "hospid mortality obs vs expected Reoperation", "Year total operation", "log obs vs expected Reop ",bin_reop_df,'total surgery count','log_Reoperation_Mort')
# ###

# df_Hosp = pd.read_csv("surgid_allyears_expec_surgid_STSRCMM_div.csv")
# plot_hosp("bin_total_cardiac", "surgid mortality total rate", "Year total operation", "mortality rate",df_Hosp,'total cardiac surgery','mort_rate_All')
# mask_bin = df_Hosp['Reop']== 0
# bin_reop_df = df_Hosp[~mask_bin]
# mask_bin = df_Hosp['FirstOperation']== 0
# bin_op_df = df_Hosp[~mask_bin]
# plot_hosp("bin_total_cardiac", "surgid mortality First operation rate", "Year total operation", "mortality first rate",bin_op_df,'total cardiac surgery','Mortality_First_rate')
# plot_hosp("bin_total_cardiac", "surgid mortality Reoperation rate", "Year total operation ", "mortality Reoperation rate",bin_reop_df,'total cardiac surgery','Mortality_Reop_rate')
#
# mask_1 = bin_op_df['Mort_observe/expected_First'] == 0
# bin_op_df = bin_op_df[~mask_1]
# mask_2 = bin_reop_df['Mort_observe/expected_Reop'] == 0
# bin_reop_df = bin_reop_df[~mask_2]
# plot_hosp("bin_total_cardiac", "surgid mortality obs vs expected First operation ", "Year total operation", "log obs vs expected first ",bin_op_df,'total cardiac surgery','log_First_Mort')
# plot_hosp("bin_total_cardiac", "surgid mortality obs vs expected Reoperation", "Year total operation", "log obs vs expected Reop ",bin_reop_df,'total cardiac surgery','log_Reoperation_Mort')
# ###

#########################################################################
# #mortality-total_CABG
# df_Hosp = pd.read_csv("hospid_allyears_expec_hospid_STSRCMM_div.csv")
# plot_hosp("bin_total_CABG", "hospid mortality total rate", "Year total operation", "mortality rate",df_Hosp,'total','mort_rate_All')
# mask_bin = df_Hosp['Reop']== 0
# bin_reop_df = df_Hosp[~mask_bin]
# mask_bin = df_Hosp['FirstOperation']== 0
# bin_op_df = df_Hosp[~mask_bin]
# plot_hosp("bin_total_CABG", "hospid mortality First operation rate", "Year total operation", "mortality first rate",bin_op_df,'total','Mortality_First_rate')
# plot_hosp("bin_total_CABG", "hospid mortality Reoperation rate", "Year total operation ", "mortality Reoperation rate",bin_reop_df,'total','Mortality_Reop_rate')
#
# mask_1 = bin_op_df['Mort_observe/expected_First'] == 0
# bin_op_df = bin_op_df[~mask_1]
# mask_2 = bin_reop_df['Mort_observe/expected_Reop'] == 0
# bin_reop_df = bin_reop_df[~mask_2]
# plot_hosp("bin_total_CABG", "hospid mortality obs vs expected First operation ", "Year total operation", "log obs vs expected first ",bin_op_df,'total','log_First_Mort')
# plot_hosp("bin_total_CABG", "hospid mortality obs vs expected Reoperation", "Year total operation", "log obs vs expected Reop ",bin_reop_df,'total','log_Reoperation_Mort')
#
# df_Hosp = pd.read_csv("surgid_allyears_expec_surgid_STSRCMM_div.csv")
# plot_hosp("bin_total_CABG", "surgsurgid mortality total rate", "Year total operation", "mortality rate",df_Hosp,'total','mort_rate_All')
# mask_bin = df_Hosp['Reop']== 0
# bin_reop_df = df_Hosp[~mask_bin]
# mask_bin = df_Hosp['FirstOperation']== 0
# bin_op_df = df_Hosp[~mask_bin]
# plot_hosp("bin_total_CABG", "surgid mortality First operation rate", "Year total operation", "mortality first rate",bin_op_df,'total','Mortality_First_rate')
# plot_hosp("bin_total_CABG", "surgid mortality Reoperation rate", "Year total operation ", "mortality Reoperation rate",bin_reop_df,'total','Mortality_Reop_rate')
#
# mask_1 = bin_op_df['Mort_observe/expected_First'] == 0
# bin_op_df = bin_op_df[~mask_1]
# mask_2 = bin_reop_df['Mort_observe/expected_Reop'] == 0
# bin_reop_df = bin_reop_df[~mask_2]
# plot_hosp("bin_total_CABG", "surgid mortality obs vs expected First operation ", "Year total operation", "log obs vs expected first ",bin_op_df,'total','log_First_Mort')
# plot_hosp("bin_total_CABG", "surgid mortality obs vs expected Reoperation", "Year total operation", "log obs vs expected Reop ",bin_reop_df,'total','log_Reoperation_Mort')


########################################################
#complics
#mortality-total_CABG
##hosp -total surgery count
##surg'total cardiac surgery'
# df_Hosp = pd.read_csv("surgid_allyears_expec_surgid_STSRCHOSPD_div.csv")
# plot_hosp("bin_total_cardiac", "surgid complics total rate", "Year total operation", "complics rate",df_Hosp,'total cardiac surgery','Complics_rate_All')
# mask_bin = df_Hosp['Reop']== 0
# bin_reop_df = df_Hosp[~mask_bin]
# mask_bin = df_Hosp['FirstOperation']== 0
# bin_op_df = df_Hosp[~mask_bin]
# plot_hosp("bin_total_cardiac", "surgid complics First operation rate", "Year total operation", "complics first rate",bin_op_df,'total cardiac surgery','Complics_First_rate')
# plot_hosp("bin_total_cardiac", "surgid complics Reoperation rate", "Year total operation ", "complics Reoperation rate",bin_reop_df,'total cardiac surgery','Complics_Reop_rate')
#
# mask_1 = bin_op_df['Comp_observe/expected_First'] == 0
# bin_op_df = bin_op_df[~mask_1]
# mask_2 = bin_reop_df['Comp_observe/expected_Reop'] == 0
# bin_reop_df = bin_reop_df[~mask_2]
# plot_hosp("bin_total_cardiac", "surgid complics obs vs expected First operation ", "Year total operation", "log obs vs expected first ",bin_op_df,'total cardiac surgery','log_First_Comp')
# plot_hosp("bin_total_cardiac", "surgid complics obs vs expected Reoperation", "Year total operation", "log obs vs expected Reop ",bin_reop_df,'total cardiac surgery','log_Reoperation_Comp')

# # #mortality-total CABG
# df_Hosp = pd.read_csv("surgid_allyears_expec_surgid_STSRCHOSPD_div.csv")
# plot_hosp("bin_total_CABG", "surgid complics total rate", "Year total operation", "complics rate",df_Hosp,'total','Complics_rate_All')
# mask_bin = df_Hosp['Reop']== 0
# bin_reop_df = df_Hosp[~mask_bin]
# mask_bin = df_Hosp['FirstOperation']== 0
# bin_op_df = df_Hosp[~mask_bin]
# plot_hosp("bin_total_CABG", "surgid complics First operation rate", "Year total operation", "complics first rate",bin_op_df,'total','Complics_First_rate')
# plot_hosp("bin_total_CABG", "surgid complics Reoperation rate", "Year total operation ", "complics Reoperation rate",bin_reop_df,'total','Complics_Reop_rate')
#
# mask_1 = bin_op_df['Mort_observe/expected_First'] == 0
# bin_op_df = bin_op_df[~mask_1]
# mask_2 = bin_reop_df['Mort_observe/expected_Reop'] == 0
# bin_reop_df = bin_reop_df[~mask_2]
# plot_hosp("bin_total_CABG", "surgid complics obs vs expected First operation ", "Year total operation", "log obs vs expected first ",bin_op_df,'total','log_First_Comp')
# plot_hosp("bin_total_CABG", "surgid complics obs vs expected Reoperation", "Year total operation", "log obs vs expected Reop ",bin_reop_df,'total','log_Reoperation_Comp')
#################################################################
#bin_Reop_CABG-mortality-surg
df_Hosp = pd.read_csv("surgid_allyears_expec_surgid_STSRCMM_div.csv")
plot_hosp("bin_Reop_CABG", "surgid mortality total rate", "Year total operation", "mortality rate",df_Hosp,'Reop','mort_rate_All')
mask_bin = df_Hosp['Reop']== 0
bin_reop_df = df_Hosp[~mask_bin]
mask_bin = df_Hosp['FirstOperation']== 0
bin_op_df = df_Hosp[~mask_bin]
plot_hosp("bin_Reop_CABG", "surgid mortality First operation rate", "Year total operation", "mortality first rate",bin_op_df,'Reop','Mortality_First_rate')
plot_hosp("bin_Reop_CABG", "surgid mortality Reoperation rate", "Year total operation ", "mortality Reoperation rate",bin_reop_df,'Reop','Mortality_Reop_rate')

mask_1 = bin_op_df['Mort_observe/expected_First'] == 0
bin_op_df = bin_op_df[~mask_1]
mask_2 = bin_reop_df['Mort_observe/expected_Reop'] == 0
bin_reop_df = bin_reop_df[~mask_2]
plot_hosp("bin_Reop_CABG", "surg mortality obs vs expected First operation ", "Year total operation", "log obs vs expected first ",bin_op_df,'Reop','log_First_Mort')
plot_hosp("bin_Reop_CABG", "surg mortality obs vs expected Reoperation", "Year total operation", "log obs vs expected Reop ",bin_reop_df,'Reop','log_Reoperation_Mort')

#bin_Reop_CABG-mortality-hosp
df_Hosp = pd.read_csv("hospid_allyears_expec_hospid_STSRCMM_div.csv")
plot_hosp("bin_Reop_CABG", "hospid mortality total rate", "Year total operation", "mortality rate",df_Hosp,'Reop','mort_rate_All')
mask_bin = df_Hosp['Reop']== 0
bin_reop_df = df_Hosp[~mask_bin]
mask_bin = df_Hosp['FirstOperation']== 0
bin_op_df = df_Hosp[~mask_bin]
plot_hosp("bin_Reop_CABG", "hospid mortality First operation rate", "Year total operation", "mortality first rate",bin_op_df,'Reop','Mortality_First_rate')
plot_hosp("bin_Reop_CABG", "hospid mortality Reoperation rate", "Year total operation ", "mortality Reoperation rate",bin_reop_df,'Reop','Mortality_Reop_rate')

mask_1 = bin_op_df['Mort_observe/expected_First'] == 0
bin_op_df = bin_op_df[~mask_1]
mask_2 = bin_reop_df['Mort_observe/expected_Reop'] == 0
bin_reop_df = bin_reop_df[~mask_2]
plot_hosp("bin_Reop_CABG", "hospid mortality obs vs expected First operation ", "Year total operation", "log obs vs expected first ",bin_op_df,'Reop','log_First_Mort')
plot_hosp("bin_Reop_CABG", "hospid mortality obs vs expected Reoperation", "Year total operation", "log obs vs expected Reop ",bin_reop_df,'Reop','log_Reoperation_Mort')

##################################3
#bin_Reop_CABG-complics-surg
# df_Hosp = pd.read_csv("surgid_allyears_expec_surgid_STSRCOM_div.csv")
# plot_hosp("bin_Reop_CABG", "surgid complics total rate", "Year total operation", "complics rate",df_Hosp,'Reop','Complics_rate_All')
# mask_bin = df_Hosp['Reop']== 0
# bin_reop_df = df_Hosp[~mask_bin]
# mask_bin = df_Hosp['FirstOperation']== 0
# bin_op_df = df_Hosp[~mask_bin]
# plot_hosp("bin_Reop_CABG", "surgid complics First operation rate", "Year total operation", "complics first rate",bin_op_df,'Reop','Complics_First_rate')
# plot_hosp("bin_Reop_CABG", "surgid complics Reoperation rate", "Year total operation ", "complics Reoperation rate",bin_reop_df,'Reop','Complics_Reop_rate')
#
# mask_1 = bin_op_df['Mort_observe/expected_First'] == 0
# bin_op_df = bin_op_df[~mask_1]
# mask_2 = bin_reop_df['Mort_observe/expected_Reop'] == 0
# bin_reop_df = bin_reop_df[~mask_2]
# plot_hosp("bin_Reop_CABG", "surg complics obs vs expected First operation ", "Year total operation", "log obs vs expected first ",bin_op_df,'Reop','log_First_Comp')
# plot_hosp("bin_Reop_CABG", "surg complics obs vs expected Reoperation", "Year total operation", "log obs vs expected Reop ",bin_reop_df,'Reop','log_Reoperation_Comp')
#
# #
#bin_Reop_CABG-complics-hosp
# df_Hosp = pd.read_csv("hospid_allyears_expec_hospid_STSRCHOSPD_div.csv")
# plot_hosp("bin_Reop_CABG", "hospid complics total rate", "Year total operation", "complics rate",df_Hosp,'Reop','Complics_rate_All')
# mask_bin = df_Hosp['Reop']== 0
# bin_reop_df = df_Hosp[~mask_bin]
# mask_bin = df_Hosp['FirstOperation']== 0
# bin_op_df = df_Hosp[~mask_bin]
# plot_hosp("bin_Reop_CABG", "hospid complics First operation rate", "Year total operation", "complics first rate",bin_op_df,'Reop','Complics_First_rate')
# plot_hosp("bin_Reop_CABG", "hospid complics Reoperation rate", "Year total operation ", "complics Reoperation rate",bin_reop_df,'Reop','Complics_Reop_rate')
#
# mask_1 = bin_op_df['Mort_observe/expected_First'] == 0
# bin_op_df = bin_op_df[~mask_1]
# mask_2 = bin_reop_df['Mort_observe/expected_Reop'] == 0
# bin_reop_df = bin_reop_df[~mask_2]
# plot_hosp("bin_Reop_CABG", "hospid complics obs vs expected First operation ", "Year total operation", "log obs vs expected first ",bin_op_df,'Reop','log_First_Comp')
# plot_hosp("bin_Reop_CABG", "hospid complics obs vs expected Reoperation", "Year total operation", "log obs vs expected Reop ",bin_reop_df,'Reop','log_Reoperation_Comp')
#
