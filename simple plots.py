import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np


def mortality_total(df,title):
    # df.plot(kind='scatter', x='total', y='mortal_prec', title=title,stacked=False)
    # plt.show()

    x = df['total_year_sum']
    y = df['mortalty_rate']
    # plt.scatter(x, y)
    df.plot(kind='scatter', x='total_year_sum', y='mortalty_rate', title=title, stacked=False)
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    plt.plot(x, p(x), "r--")

    plt.show()


def mortality_total_surg(df,title):
    # df.plot(kind='scatter', x='total', y='mortal_prec', title=title,stacked=False)
    # plt.show()

    x = df['total_year_count']
    y = df['mortalty_rate']
    # plt.scatter(x, y)
    df.plot(kind='scatter', x='total_year_count', y='mortalty_rate', title=title, stacked=False)
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    plt.plot(x, p(x), "r--")

    plt.show()


def draw_hist_c(data,num_of_bins,title,x_title,y_title):
    plt.hist(data, bins=num_of_bins, color="plum",ec="black")
    plt.title(title)
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.show()

def draw_hist(data,num_of_bins,title,x_title,y_title,color):
    plt.hist(data, bins=num_of_bins, color=color,ec="black")
    plt.title(title)
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.show()

path="/tmp/pycharm_project_723/"


df_avg_siteid = pd.read_csv("total_avg_site_id.csv")
df_avg_surgid = pd.read_csv("total_avg_surgid.csv")

# # df_sum_hospid= pd.read_csv(path+"sum all years hospid.csv")
# draw_hist(df_avg_siteid['total_year_avg'],40,"siteid Histogram of yearly avg operation",'avg of Operation',"count of siteid",'skyblue')
# draw_hist(df_avg_siteid['Year_avg_Firstop'],40,"siteid Histogram of yearly avg First operation",'avg of First Operation',"count of siteid",'skyblue')
# draw_hist(df_avg_siteid['Year_avg_reop'],40,"siteid Histogram of yearly avg reOperation",'avg of reOperation',"count of siteid",'skyblue')
#
# draw_hist(df_avg_siteid['firstop/total'],40,"siteid Histogram of yearly avg First operation/Total operation",'% of First Operation',"count of siteid",'palegreen')
# draw_hist(df_avg_siteid['reop/total'],40,"siteid Histogram of yearly avg reOperation/Total operation",'% of reOperation',"count of siteid",'palegreen')
#
# # draw_hist(df_sum_surgid['Year_avg'],20,"surgid Histogram of yearly avg operation",'avg of Operation',"count of surgid")
# draw_hist(df_avg_surgid['total_year_avg'],40,"surgid Histogram of yearly avg operation",'avg of Operation',"count of surgid",'plum')
# draw_hist(df_avg_surgid['Year_avg_Firstop'],40,"surgid Histogram of yearly avg First operation",'avg of First Operation',"count of surgid",'plum')
# draw_hist(df_avg_surgid['Year_avg_reop'],40,"surgid Histogram of yearly avg reOperation",'avg of reOperation',"count of surgid",'plum')
#
# draw_hist(df_avg_surgid['firstop/total'],40,"surgid Histogram of yearly avg First operation/Total operation",'% of First Operation',"count of surgid",'bisque')
# draw_hist(df_avg_surgid['reop/total'],40,"surgid Histogram of yearly avg reOperation/Total operation",'% of reOperation',"count of surgid",'bisque')

# mortality_total(df_avg_siteid, " site id: mortality - total ops")
#
# mortality_total_surg(df_avg_surgid, " surgeon id: mortality - total ops")


# ax = plt.gca()
#
# ax.scatter(df_avg_siteid['total_year_avg'], df_avg_siteid['mortalty_reop_rate'], color="lightcoral")
# ax.scatter(df_avg_siteid['total_year_avg'], df_avg_siteid['Complics_reop_rate'], color="lightseagreen")
# f = lambda c : plt.plot([],color=c, ls="", marker="o")[0]
# ax.legend(handles = [f("lightcoral"), f("lightseagreen")],
#            labels=['mortalty', 'Complics'])
# plt.title("SiteID Yearly average")
# plt.xlabel("count of operations")
# plt.ylabel("mortality or complics rate")
# plt.show()


def yearly_avg_siteid():
    ax = plt.gca()

    ax.scatter(df_avg_siteid['total_year_avg'], df_avg_siteid['mortalty_reop_rate'], color="palevioletred",s=30)
    ax.scatter(df_avg_siteid['total_year_avg'], df_avg_siteid['Complics_reop_rate'], color="darkturquoise",edgecolor='lightseagreen',s=30)

    plt.title("SiteID yearly average")
    plt.xlabel("yearly average of operations")
    plt.ylabel("mortality or complication reop rate")

    x = df_avg_siteid['total_year_avg']
    y = df_avg_siteid['mortalty_reop_rate']
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    plt.plot(x, p(x), "mediumvioletred")

    a = df_avg_siteid['total_year_avg']
    b = df_avg_siteid['Complics_reop_rate']
    c = np.polyfit(a, b, 1)
    t = np.poly1d(c)
    text = f" Mortality : $Y={z[0]:0.6f}X{z[1]:+0.6f}$"  # \n$R^2 = {r2_score(y, p):0.3f}$"
    text2 = f" Complication : $Y={c[0]:0.6f}X{c[1]:+0.6f}$"
    f = lambda c: plt.plot([], color=c, ls="", marker="o")[0]
    ax.legend(handles=[f("palevioletred"), f("mediumturquoise")],
              labels=[text, text2])
    plt.plot(a, t(a), "teal")
    plt.savefig('SiteID yearly average.png')

    plt.show()

def yearly_First_operation_siteid():
    ax = plt.gca()

    ax.scatter(df_avg_siteid['Year_avg_Firstop'], df_avg_siteid['mortalty_reop_rate'], color="palevioletred", s=30)
    ax.scatter(df_avg_siteid['Year_avg_Firstop'], df_avg_siteid['Complics_reop_rate'],color="darkturquoise",edgecolor='lightseagreen', s=30)

    plt.title("SiteID yearly average for first operation")
    plt.xlabel("yearly average of first operations")
    plt.ylabel("mortality or complication reop rate")

    x = df_avg_siteid['Year_avg_Firstop']
    y = df_avg_siteid['mortalty_reop_rate']
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    plt.plot(x, p(x), "mediumvioletred")

    a = df_avg_siteid['Year_avg_Firstop']
    b = df_avg_siteid['Complics_reop_rate']
    c = np.polyfit(a, b, 1)
    t = np.poly1d(c)
    text = f" Mortality : $Y={z[0]:0.6f}X{z[1]:+0.6f}$"  # \n$R^2 = {r2_score(y, p):0.3f}$"
    text2 = f" Complication : $Y={c[0]:0.6f}X{c[1]:+0.6f}$"
    f = lambda c: plt.plot([], color=c, ls="", marker="o")[0]
    ax.legend(handles=[f("palevioletred"), f("mediumturquoise")],
              labels=[text, text2])
    plt.plot(a, t(a), "teal")
    plt.savefig('SiteID yearly average for first operation.png')
    plt.show()

def yearly_reoperation_siteid():
    mask = df_avg_siteid['Year_sum_reop'] == 0
    df_reop = df_avg_siteid[~mask]
    ax = plt.gca()

    ax.scatter(df_reop['Year_avg_reop'], df_reop['mortalty_reop_rate'], color="palevioletred", s=30)
    ax.scatter(df_reop['Year_avg_reop'], df_reop['Complics_reop_rate'], color="darkturquoise",edgecolor='lightseagreen',s=30)
    # f = lambda c: plt.plot([], color=c, ls="", marker="o")[0]
    # ax.legend(handles=[f("palevioletred"), f("mediumturquoise")],
    #           labels=['mortalty', 'Complication'])
    plt.title("SiteID yearly average for Reoperation")
    plt.xlabel("yearly average of Reoperation")
    plt.ylabel("mortality or complication reop rate")

    x= df_reop['Year_avg_reop']
    y= df_reop['mortalty_reop_rate']
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    plt.plot(x, p(x), "mediumvioletred")

    a = df_reop['Year_avg_reop']
    b = df_reop['Complics_reop_rate']
    c = np.polyfit(a, b, 1)
    t = np.poly1d(c)
    text = f" Mortality : $Y={z[0]:0.6f}X{z[1]:+0.6f}$"  # \n$R^2 = {r2_score(y, p):0.3f}$"
    text2 = f" Complication : $Y={c[0]:0.6f}X{c[1]:+0.6f}$"
    f = lambda c: plt.plot([], color=c, ls="", marker="o")[0]
    ax.legend(handles=[f("palevioletred"), f("mediumturquoise")],
              labels=[text, text2])
    plt.plot(a, t(a), "teal")
    plt.savefig('SiteID yearly average for Reoperation.png')
    plt.show()

def yearly_avg_surgid():
    ax = plt.gca()

    ax.scatter(df_avg_surgid['total_year_avg'], df_avg_surgid['mortalty_reop_rate'], color="orchid", s=30)
    ax.scatter(df_avg_surgid['total_year_avg'], df_avg_surgid['Complics_reop_rate'], color="steelblue",edgecolor='tab:blue',s=30)
    # f = lambda c: plt.plot([], color=c, ls="", marker="o")[0]
    # ax.legend(handles=[f("orchid"), f("steelblue")],
    #           labels=['mortalty', 'Complication'])
    plt.title("Surgid yearly average")
    plt.xlabel("yearly average of operations")
    plt.ylabel("mortality or Complication reop rate")

    x = df_avg_surgid['total_year_avg']
    y = df_avg_surgid['mortalty_reop_rate']
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    plt.plot(x, p(x), "darkorchid")

    a = df_avg_surgid['total_year_avg']
    b = df_avg_surgid['Complics_reop_rate']
    c = np.polyfit(a, b, 1)
    t = np.poly1d(c)
    plt.plot(a, t(a), "mediumblue")
    text = f" Mortality : $Y={z[0]:0.6f}X{z[1]:+0.6f}$"  # \n$R^2 = {r2_score(y, p):0.3f}$"
    text2 = f" Complication : $Y={c[0]:0.6f}X{c[1]:+0.6f}$"
    f = lambda c: plt.plot([], color=c, ls="", marker="o")[0]
    ax.legend(handles=[f("orchid"), f("steelblue")],
              labels=[text, text2])
    plt.savefig('Surgid yearly average.png')
    plt.show()

def yearly_avg_First_operation_surgid():
    ax = plt.gca()

    ax.scatter(df_avg_surgid['Year_avg_Firstop'], df_avg_surgid['mortalty_reop_rate'], color="orchid", s=30)
    ax.scatter(df_avg_surgid['Year_avg_Firstop'], df_avg_surgid['Complics_reop_rate'], color="steelblue", edgecolor='tab:blue', s=30)
    # f = lambda c: plt.plot([], color=c, ls="", marker="o")[0]
    # ax.legend(handles=[f("orchid"), f("steelblue")],
    #           labels=['mortalty', 'Complication'])
    plt.title("Surgid yearly average for first operation")
    plt.xlabel("yearly average of first operations")
    plt.ylabel("mortality or Complication reop rate")

    x = df_avg_surgid['Year_avg_Firstop']
    y = df_avg_surgid['mortalty_reop_rate']
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    plt.plot(x, p(x), "darkorchid")

    a = df_avg_surgid['Year_avg_Firstop']
    b = df_avg_surgid['Complics_reop_rate']
    c = np.polyfit(a, b, 1)
    t = np.poly1d(c)
    plt.plot(a, t(a), "mediumblue")
    text = f" Mortality : $Y={z[0]:0.6f}X{z[1]:+0.6f}$"  # \n$R^2 = {r2_score(y, p):0.3f}$"
    text2 = f" Complication : $Y={c[0]:0.6f}X{c[1]:+0.6f}$"
    f = lambda c: plt.plot([], color=c, ls="", marker="o")[0]
    ax.legend(handles=[f("orchid"), f("steelblue")],
              labels=[text, text2])
    plt.savefig('Surgid yearly average for first operation.png')
    plt.show()

def yearly_avg_reoperation_surgid():
    mask = df_avg_surgid['Year_sum_reop'] == 0
    df_reop = df_avg_surgid[~mask]
    ax = plt.gca()

    ax.scatter(df_reop['Year_avg_reop'], df_reop['mortalty_reop_rate'], color="orchid", s=30)
    ax.scatter(df_reop['Year_avg_reop'], df_reop['Complics_reop_rate'], color="steelblue", edgecolor='tab:blue', s=30)
    # f = lambda c: plt.plot([], color=c, ls="", marker="o")[0]
    # ax.legend(handles=[f("orchid"), f("steelblue")],
    #           labels=['mortalty', 'Complication'])
    plt.title("Surgid yearly average for Reoperation")
    plt.xlabel("yearly average of reoperations")
    plt.ylabel("mortality or Complication reop rate")

    x = df_reop['Year_avg_reop']
    y = df_reop['mortalty_reop_rate']
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    plt.plot(x, p(x), "darkorchid")

    a = df_reop['Year_avg_reop']
    b = df_reop['Complics_reop_rate']
    c = np.polyfit(a, b, 1)
    t = np.poly1d(c)
    plt.plot(a, t(a), "mediumblue")
    text = f" Mortality : $Y={z[0]:0.6f}X{z[1]:+0.6f}$"  # \n$R^2 = {r2_score(y, p):0.3f}$"
    text2 = f" Complication : $Y={c[0]:0.6f}X{c[1]:+0.6f}$"
    f = lambda c: plt.plot([], color=c, ls="", marker="o")[0]
    ax.legend(handles=[f("orchid"), f("steelblue")],
              labels=[text, text2])
    plt.savefig('Surgid yearly average for Reoperation.png')
    plt.show()

def pearson_correlation_siteid():
    pearson_corr_siteid = df_avg_siteid[
        ['total_year_avg', 'Year_avg_Firstop', 'Year_avg_reop', 'reop/total', 'mortalty_rate']]
    correlation = pearson_corr_siteid.corr(method='spearman')
    # fig, ax = plt.subplots(figsize=(15, 15))
    # sb.heatmap(correlation, annot=True, linewidths=1, cmap='coolwarm', square=True, ax=ax)
    # plt.show()

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title('Spearman correlation for siteid variables')
    sb.heatmap(correlation,
               xticklabels=correlation.columns,
               yticklabels=correlation.columns,
               cmap='viridis',
               annot=True,
               fmt='f',
               linewidth=0.5, ax=ax)
    # plt.show()

    g = sb.PairGrid(pearson_corr_siteid, diag_sharey=False, corner=True)
    g.map_lower(sb.regplot,color=".5")
    g.map_diag(sb.histplot)
    g.fig.suptitle('Pairwise plots for siteid correlation')

    # g = sb.PairGrid(pearson_corr_siteid)
    # g.map_diag(sb.histplot)
    # g.map_upper(sb.regplot, color=".5")
    # g.map_lower(sb.scatterplot,color=".3")
    # g.fig.suptitle('Pairwise plots for siteid correlation')
    plt.show()

def pearson_correlation_surgid():
    pearson_corr_surgid = df_avg_surgid[['total_year_avg', 'Year_avg_Firstop', 'Year_avg_reop','reop/total','mortalty_rate']]
    correlation = pearson_corr_surgid.corr(method='spearman')
    # fig, ax = plt.subplots(figsize=(15, 15))
    # sb.heatmap(correlation, annot=True, linewidths=1, cmap='coolwarm', square=True, ax=ax)
    # plt.show()
    print(correlation)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title('Spearman correlation for surgid variables')
    sb.heatmap(correlation,
               xticklabels=correlation.columns,
               yticklabels=correlation.columns,
               cmap='viridis',
               annot=True,
               fmt='f',
               linewidth=0.5, ax=ax)

    g = sb.PairGrid(pearson_corr_surgid, diag_sharey=False, corner=True)
    g.map_lower(sb.regplot,color=".5")
    g.map_diag(sb.histplot)
    g.fig.suptitle('Pairwise plots for surgid correlation')

    # g = sb.PairGrid(pearson_corr_surgid)
    # g.map_diag(sb.histplot)
    # g.map_upper(sb.regplot, color=".5")
    # g.map_lower(sb.scatterplot,color=".3")
    # g.fig.suptitle('Pairwise plots for surgid correlation')
    plt.show()

yearly_avg_siteid()
yearly_First_operation_siteid()
yearly_reoperation_siteid()

yearly_avg_surgid()
yearly_avg_First_operation_surgid()
yearly_avg_reoperation_surgid()
#
#
# pearson_correlation_siteid()
# pearson_correlation_surgid()



# df= pd.read_csv("total_avg_site_id.csv")
# # total_year_sum
# new_df=pd.DataFrame(data=df,columns=['mortalty_reop_rate','total_year_avg'])
# print(new_df)
#
# new_df.to_csv("box.xls")
# new_df['total operations'] = pd.qcut(new_df['total_year_avg'], 4, labels=['I', 'II', 'III', 'IV'])
# bp = new_df.boxplot(column='mortalty_reop_rate', by='total operations',patch_artist=True)
#
# for patch in bp['boxes']:
#     patch.set(facecolor='cyan')
# # colors = [ 'lightblue', 'lightgreen', 'tan', 'pink']
# # for patch, color in zip(box['boxes'], colors):
# #     patch.set_facecolor(color)
#
# # plt.show()
# f = lambda c: plt.plot([], color=c, ls="", marker="o")[0]
# plt.legend(handles=[f("palevioletred"), f("mediumturquoise")],
#               labels=['mortalty', 'Complics'])
#
# plt.show()
#
#
# # def box_plot(data, edge_color, fill_color):
# #     bp = ax.boxplot(data, patch_artist=True)
# #
# #     for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
# #         plt.setp(bp[element], color=edge_color)
# #
# #     for patch in bp['boxes']:
# #         patch.set(facecolor=fill_color)
# # #
# #
# example_data1 = [[1, 2, 0.8], [0.5, 2, 2], [3, 2, 1]]
# # example_data2 = [[5, 3, 4], [6, 4, 3, 8], [6, 4, 9]]
# #
# fig, ax = plt.subplots()
# bp = ax.boxplot(example_data1, patch_artist=True)
# for patch in bp['boxes']:
#     patch.set(facecolor='cyan')
# # box_plot(example_data1, 'black', 'tan')
# # box_plot(example_data2, 'black', 'cyan')
# ax.set_ylim(0, 10)
# plt.show()

# df= pd.read_csv("total_avg_surgid.csv")
# # total_year_sum
# new_df=pd.DataFrame(data=df,columns=['mortalty_reop_rate','total_year_avg'])
#
# new_df['bins'] = pd.qcut(new_df['total_year_avg'], 3, labels=['low', 'mid', 'high'])
# print(new_df)
# new_df.to_csv("box.csv")
# f = lambda c: plt.plot([], color=c, ls="", marker="o")[0]
# new_df.boxplot(column='mortalty_reop_rate', by='bins')
# plt.legend(handles=[f("white"), f("white"), f("white")],
#               labels=['low 1-32.7', 'mid 32.75-61.286','high 61.3-339.5'])
# plt.ylabel("test")
# plt.show()
