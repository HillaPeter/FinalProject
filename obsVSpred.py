import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import r2_score
import scipy.stats

df_mort_siteid = pd.read_csv("mortality siteid obs vs expected.csv")
df_mort_surgid = pd.read_csv("mortality surgid obs vs expected.csv")
df_complics_siteid = pd.read_csv("Complics siteid obs vs expected.csv")
df_complics_surgid = pd.read_csv("Complics surgid obs vs expected.csv")

df = pd.read_csv("total_avg_surgid.csv")
df1 = pd.read_csv("total_avg_site_id.csv")

def siteid_obs_vs_expected_mort():
    mask = df_mort_siteid['count_Reop'] == 0
    df_reop = df_mort_siteid[~mask]
    mask = df_mort_siteid['count_First'] == 0
    df_op = df_mort_siteid[~mask]
    ax = plt.gca()

    ax.scatter(df_op['Year_avg_Firstop'], df_op['log_First'], color="plum",edgecolor='orchid', s=30)
    # ax.scatter(df_reop['Year_avg_reop'], df_reop['log_Reoperation'], color="lightskyblue", edgecolor='tab:blue', s=30)

    plt.title('Siteid observe vs expected Mortality First operation')

    plt.xlabel("Yearly AVG of first operation")
    plt.ylabel("mortality observe vs expected of first operation")

    x = df_op['Year_avg_Firstop']
    y = df_op['log_First']
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    plt.plot(x, p(x), "purple")


    # a = df_reop['Year_avg_reop']
    # b = df_reop['log_Reoperation']
    # c = np.polyfit(a, b, 1)
    # t = np.poly1d(c)
    # plt.plot(a, t(a), "mediumblue")
    text = f" First : $Y={z[0]:0.6f}X{z[1]:+0.6f}$"  # \n$R^2 = {r2_score(y, p):0.3f}$"
    # text2 = f" Reoperation : $Y={c[0]:0.6f}X{c[1]:+0.6f}$"
    # r, p = scipy.stats.spearmanr(a, b)
    r1, p1 = scipy.stats.spearmanr(x, y)
    text3 = f" Spearman Corr : {r1:0.4f}   P-value : {p1:0.4f}"
    # text4 = f" Spearman Corr : {r:0.4f}   P-value : {p:0.4f}"

    print(text)
    # print(text2)
    f = lambda c: plt.plot([], color=c, ls="", marker="o")[0]
    ax.legend(handles=[f("orchid"), f("white")],
              labels=[text, text3])
    # fig = plt.figure()
    # fig.subplots_adjust()
    # plt.text(30, 50, text)
    # plt.text(90, 45, text2)
    # plt.title("y=%.6fx^2+%.6fx+(%.6f)" % (z[0], z[1], z[2]))
    # print("y=%.6fx^2+%.6fx+(%.6f)" % (z[0], z[1], z[2]))
    # plt.savefig('Surgid yearly average for Reoperation.png')
    # plt.gca().text(0.05, 0.95, text, transform=plt.gca().transAxes,
    #                fontsize=14, verticalalignment='top')
    plt.show()


def siteid_obs_vs_expected_mort_reop():
    mask = df_mort_siteid['count_Reop'] == 0
    df_reop = df_mort_siteid[~mask]

    ax = plt.gca()


    ax.scatter(df_reop['Year_avg_reop'], df_reop['log_Reoperation'], color="lightskyblue", edgecolor='tab:blue', s=30)

    plt.title('Siteid observe vs expected Mortality Reoperation')

    plt.xlabel("Yearly AVG of Reoperation")
    plt.ylabel("mortality observe vs expected Reoperation")
    a = df_reop['Year_avg_reop']
    b = df_reop['log_Reoperation']
    c = np.polyfit(a, b, 1)
    t = np.poly1d(c)
    plt.plot(a, t(a), "mediumblue")
    r, p = scipy.stats.spearmanr(a, b)

    text2 = f" Reoperation : $Y={c[0]:0.6f}X{c[1]:+0.6f}$"
    text3 = f" Spearman Corr : {r:0.4f}   P-value : {p:0.4f}"
    print(text2)
    f = lambda c: plt.plot([], color=c, ls="", marker="o")[0]
    ax.legend(handles=[ f("lightskyblue"),f("white")],
              labels=[ text2,text3])
    plt.show()

def surgid_obs_vs_expected_mort():
    mask = df_mort_surgid['count_Reop'] == 0
    df_reop = df_mort_surgid[~mask]
    mask = df_mort_surgid['count_First'] == 0
    df_op = df_mort_surgid[~mask]
    ax = plt.gca()

    ax.scatter(df_op['Year_avg_Firstop'], df_op['log_First'], color="plum",edgecolor='orchid', s=30)
    ax.scatter(df_reop['Year_avg_reop'], df_reop['log_Reoperation'], color="lightskyblue", edgecolor='tab:blue', s=30)

    plt.title('Surgid observe vs expected Mortality')

    plt.xlabel("Yearly AVG of operation")
    plt.ylabel("mortality observe vs expected")

    x = df_op['Year_avg_Firstop']
    y = df_op['log_First']
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    plt.plot(x, p(x), "purple")

    a = df_reop['Year_avg_reop']
    b = df_reop['log_Reoperation']
    c = np.polyfit(a, b, 1)
    t = np.poly1d(c)
    plt.plot(a, t(a), "mediumblue")
    text = f" First : $Y={z[0]:0.6f}X{z[1]:+0.6f}$" #\n$R^2 = {r2_score(y, p):0.3f}$"
    text2 = f" Reoperation : $Y={c[0]:0.6f}X{c[1]:+0.6f}$"

    r, p = scipy.stats.spearmanr(a, b)
    r1, p1 = scipy.stats.spearmanr(x, y)
    text3 = f" Spearman Corr : {r1:0.4f}   P-value : {p1:0.4f}"
    text4 = f" Spearman Corr : {r:0.4f}   P-value : {p:0.4f}"

    print (text)
    print(text2)
    f = lambda c: plt.plot([], color=c, ls="", marker="o")[0]
    ax.legend(handles=[f("orchid"),f("white"), f("steelblue"),f("white")],
              labels=[text,text3, text2,text4])
    # ax.text(right, top, 'right top',
    #         horizontalalignment='right',
    #         verticalalignment='top',
    #         transform=ax.transAxes)

    # plt.text(100, 50, text)
    # plt.text(100, 45, text2)
    # plt.title("y=%.6fx^2+%.6fx+(%.6f)" % (z[0], z[1], z[2]))
    # print("y=%.6fx^2+%.6fx+(%.6f)" % (z[0], z[1], z[2]))
    # plt.savefig('Surgid yearly average for Reoperation.png')
    # plt.gca().text(0.05, 0.95, text, transform=plt.gca().transAxes,
    #                fontsize=14, verticalalignment='top')
    plt.show()

def siteid_obs_vs_expected_complics():
    mask = df_complics_siteid['count_Reop'] == 0
    df_reop = df_complics_siteid[~mask]
    mask = df_complics_siteid['count_First'] == 0
    df_op = df_complics_siteid[~mask]
    ax = plt.gca()

    ax.scatter(df_op['Year_avg_Firstop'], df_op['log_First'], color="palevioletred",edgecolor='indianred', s=30)
    ax.scatter(df_reop['Year_avg_reop'], df_reop['log_Reoperation'], color="darkturquoise",edgecolor='lightseagreen',s=30)

    plt.title('Siteid observe vs expected Complications')

    plt.xlabel("Yearly AVG of operation")
    plt.ylabel("complication observe vs expected")

    x = df_op['Year_avg_Firstop']
    y = df_op['log_First']
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    plt.plot(x, p(x), "maroon")

    a = df_reop['Year_avg_reop']
    b = df_reop['log_Reoperation']
    c = np.polyfit(a, b, 1)
    t = np.poly1d(c)
    plt.plot(a, t(a), "darkgreen")
    text = f" First : $Y={z[0]:0.6f}X{z[1]:+0.6f}$"  # \n$R^2 = {r2_score(y, p):0.3f}$"
    text2 = f" Reoperation : $Y={c[0]:0.6f}X{c[1]:+0.6f}$"
    r, p = scipy.stats.spearmanr(a, b)
    r1, p1 = scipy.stats.spearmanr(x, y)
    text3 = f" Spearman Corr : {r1:0.4f}   P-value : {p1:0.4f}"
    text4 = f" Spearman Corr : {r:0.4f}   P-value : {p:0.4f}"

    f = lambda c: plt.plot([], color=c, ls="", marker="o")[0]
    ax.legend(handles=[f("palevioletred"),f("white"), f("darkturquoise"),f("white")],
              labels=[text,text3, text2,text4])
    # fig = plt.figure()
    # fig.subplots_adjust()
    # plt.text(30, 50, text)
    # plt.text(90, 45, text2)
    # plt.title("y=%.6fx^2+%.6fx+(%.6f)" % (z[0], z[1], z[2]))
    # print("y=%.6fx^2+%.6fx+(%.6f)" % (z[0], z[1], z[2]))
    # plt.savefig('Surgid yearly average for Reoperation.png')
    # plt.gca().text(0.05, 0.95, text, transform=plt.gca().transAxes,
    #                fontsize=14, verticalalignment='top')
    plt.show()


def surgid_obs_vs_expected_complics():
    mask = df_complics_surgid['count_Reop'] == 0
    df_reop = df_complics_surgid[~mask]
    mask = df_complics_surgid['count_First'] == 0
    df_op = df_complics_surgid[~mask]
    ax = plt.gca()

    ax.scatter(df_op['Year_avg_Firstop'], df_op['log_First'], color="palevioletred",edgecolor='indianred',s=30)
    ax.scatter(df_reop['Year_avg_reop'], df_reop['log_Reoperation'], color="darkturquoise",edgecolor='lightseagreen',s=30)

    plt.title('Surgid observe vs expected Complications')

    plt.xlabel("Yearly AVG of operation")
    plt.ylabel("complication observe vs expected")

    x = df_op['Year_avg_Firstop']
    y = df_op['log_First']
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    plt.plot(x, p(x), "maroon")

    a = df_reop['Year_avg_reop']
    b = df_reop['log_Reoperation']
    c = np.polyfit(a, b, 1)
    t = np.poly1d(c)
    plt.plot(a, t(a), "darkgreen")
    text = f" First : $Y={z[0]:0.6f}X{z[1]:+0.6f}$"  # \n$R^2 = {r2_score(y, p):0.3f}$"
    text2 = f" Reoperation : $Y={c[0]:0.6f}X{c[1]:+0.6f}$"
    print(text)
    print(text2)
    r, p = scipy.stats.spearmanr(a, b)
    r1, p1 = scipy.stats.spearmanr(x, y)
    text3 = f" Spearman Corr : {r1:0.4f}   P-value : {p1:0.4f}"
    text4 = f" Spearman Corr : {r:0.4f}   P-value : {p:0.4f}"

    f = lambda c: plt.plot([], color=c, ls="", marker="o")[0]
    ax.legend(handles=[f("palevioletred"), f("white"), f("darkturquoise"), f("white")],
              labels=[text, text3, text2, text4])
    # fig = plt.figure()
    # fig.subplots_adjust()
    # plt.text(30, 50, text)
    # plt.text(90, 45, text2)
    # plt.title("y=%.6fx^2+%.6fx+(%.6f)" % (z[0], z[1], z[2]))
    # print("y=%.6fx^2+%.6fx+(%.6f)" % (z[0], z[1], z[2]))
    # plt.savefig('Surgid yearly average for Reoperation.png')
    # plt.gca().text(0.05, 0.95, text, transform=plt.gca().transAxes,
    #                fontsize=14, verticalalignment='top')
    plt.show()


def mortality_reop_surgid_boxplot():
    mask = df['Year_sum_reop'] == 0
    df_reop = df[~mask]
    # total_year_sum
    new_df=pd.DataFrame(data=df_reop,columns=['mortalty_reop_rate','total_year_avg'])

    new_df['bins'] = pd.qcut(new_df['total_year_avg'], 3, labels=['low', 'mid', 'high'])
    print(new_df)
    new_df.to_csv("box_surgid_mort.csv")
    mask = new_df['bins'] == 'low'
    df_low = new_df[mask]
    mask = new_df['bins'] == 'mid'
    df_mid = new_df[mask]
    mask = new_df['bins'] == 'high'
    df_high = new_df[mask]

    data = [df_low['mortalty_reop_rate'],df_mid['mortalty_reop_rate'],df_high['mortalty_reop_rate']]
    print (df_low.describe())
    print(df_mid.describe())
    print(df_high.describe())
    text = f" low\n ${df_low['total_year_avg'].min(): 0.2f} - ${df_low['total_year_avg'].max(): 0.2f}\n   Mean = ${df_low['mortalty_reop_rate'].mean():0.6f} $"
    text2 = f"mid\n ${df_mid['total_year_avg'].min(): 0.2f} - ${df_mid['total_year_avg'].max(): 0.2f}\n   Mean = ${df_mid['mortalty_reop_rate'].mean():0.6f} $"
    text3 =f"high\n${df_high['total_year_avg'].min(): 0.2f} - ${df_high['total_year_avg'].max(): 0.2f}\n Mean = ${df_high['mortalty_reop_rate'].mean():0.6f} $"
    # ax = plt.gca()
    # ax = sns.boxplot(x="day", y="total_bill", data=df_mid['mortalty_reop_rate'])
    # show plot
    labels = [text,text2,text3]


    fig1, ax1 = plt.subplots()
    ax1.set_title('Mortality surgid reop boxplot')
    bp = ax1.boxplot(data, patch_artist=True, labels=labels)
    colors = ['pink', 'lightblue', 'palegreen']
    for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)

    f = lambda c: plt.plot([], color=c, ls="", marker="o")[0]
    plt.legend(handles=[f("pink"), f("lightblue"), f("palegreen")],
               labels=['low', 'mid', 'high'])
    plt.ylabel("Mortality Reop rate")
    plt.show()


    # ax = plt.gca()
    #
    # f = lambda c: plt.plot([], color=c, ls="", marker="o")[0]
    # new_df.boxplot(column='mortalty_reop_rate', by='bins')
    # plt.legend(handles=[f("palevioletred"), f("mediumturquoise"), f("yellow")],
    #               labels=['low', 'mid','high'])
    #
    # plt.show()

def mortality_reop_siteid_boxplot():
    mask = df1['Year_sum_reop'] == 0
    df_reop = df1[~mask]
    # total_year_sum
    new_df=pd.DataFrame(data=df_reop,columns=['mortalty_reop_rate','total_year_avg'])

    new_df['bins'] = pd.qcut(new_df['total_year_avg'], 3, labels=['low', 'mid', 'high'])
    print(new_df)
    new_df.to_csv("box_siteid_mort.csv")
    mask = new_df['bins'] == 'low'
    df_low = new_df[mask]
    mask = new_df['bins'] == 'mid'
    df_mid = new_df[mask]
    mask = new_df['bins'] == 'high'
    df_high = new_df[mask]

    data = [df_low['mortalty_reop_rate'],df_mid['mortalty_reop_rate'],df_high['mortalty_reop_rate']]
    print (df_low.describe())
    print(df_mid.describe())
    print(df_high.describe())
    text = f" low\n ${df_low['total_year_avg'].min(): 0.2f} - ${df_low['total_year_avg'].max(): 0.2f}\n  Mean = ${df_low['mortalty_reop_rate'].mean():0.6f} $"
    text2 = f"mid\n ${df_mid['total_year_avg'].min(): 0.2f} - ${df_mid['total_year_avg'].max(): 0.2f}\n   Mean = ${df_mid['mortalty_reop_rate'].mean():0.6f} $"
    text3 = f"high\n${df_high['total_year_avg'].min(): 0.2f} - ${df_high['total_year_avg'].max(): 0.2f}\n Mean = ${df_high['mortalty_reop_rate'].mean():0.6f} $"
    labels = [text, text2, text3]



    fig1, ax1 = plt.subplots()
    ax1.set_title('Mortality siteid reop boxplot')
    bp = ax1.boxplot(data, patch_artist=True, labels=labels)
    colors = ['pink', 'lightblue', 'palegreen']
    for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)

    f = lambda c: plt.plot([], color=c, ls="", marker="o")[0]
    plt.legend(handles=[f("pink"), f("lightblue"), f("palegreen")],
               labels=['low', 'mid', 'high'])
    plt.ylabel("Mortality Reop rate")
    plt.show()

def complics_reop_surgid_boxplot():
    mask = df['Year_sum_reop'] == 0
    df_reop = df[~mask]
    # total_year_sum
    new_df=pd.DataFrame(data=df_reop,columns=['Complics_reop_rate','total_year_avg'])

    new_df['bins'] = pd.qcut(new_df['total_year_avg'], 3, labels=['low', 'mid', 'high'])
    print(new_df)
    new_df.to_csv("box_surgid_complics.csv")
    mask = new_df['bins'] == 'low'
    df_low = new_df[mask]
    mask = new_df['bins'] == 'mid'
    df_mid = new_df[mask]
    mask = new_df['bins'] == 'high'
    df_high = new_df[mask]

    data = [df_low['Complics_reop_rate'],df_mid['Complics_reop_rate'],df_high['Complics_reop_rate']]
    print (df_low.describe())
    print(df_mid.describe())
    print(df_high.describe())
    text = f" low\n${df_low['total_year_avg'].min(): 0.2f} - ${df_low['total_year_avg'].max(): 0.2f}\n  Mean = ${df_low['Complics_reop_rate'].mean():0.6f} $"
    text2 = f"mid\n${df_mid['total_year_avg'].min(): 0.2f} - ${df_mid['total_year_avg'].max(): 0.2f}\n   Mean = ${df_mid['Complics_reop_rate'].mean():0.6f} $"
    text3 =f"high\n${df_high['total_year_avg'].min(): 0.2f} - ${df_high['total_year_avg'].max(): 0.2f}\n Mean = ${df_high['Complics_reop_rate'].mean():0.6f} $"
    labels = [text, text2, text3]

    fig1, ax1 = plt.subplots()
    ax1.set_title('Complication surgid reop boxplot')
    bp = ax1.boxplot(data, patch_artist=True, labels=labels)
    colors = ['pink', 'lightblue', 'palegreen']
    for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)

    f = lambda c: plt.plot([], color=c, ls="", marker="o")[0]
    plt.legend(handles=[f("pink"), f("lightblue"), f("palegreen")],
               labels=['low', 'mid', 'high'])
    plt.ylabel("Complication Reop rate")
    plt.show()

def complics_reop_siteid_boxplot():
    mask = df1['Year_sum_reop'] == 0
    df_reop = df1[~mask]
    # total_year_sum
    new_df=pd.DataFrame(data=df_reop,columns=['Complics_reop_rate','total_year_avg'])

    new_df['bins'] = pd.qcut(new_df['total_year_avg'], 3, labels=['low', 'mid', 'high'])
    print(new_df)
    new_df.to_csv("box_sitrid_complics.csv")
    mask = new_df['bins'] == 'low'
    df_low = new_df[mask]
    mask = new_df['bins'] == 'mid'
    df_mid = new_df[mask]
    mask = new_df['bins'] == 'high'
    df_high = new_df[mask]

    data = [df_low['Complics_reop_rate'],df_mid['Complics_reop_rate'],df_high['Complics_reop_rate']]
    print (df_low.describe())
    print(df_mid.describe())
    print(df_high.describe())
    text = f" low\n ${df_low['total_year_avg'].min(): 0.2f} - ${df_low['total_year_avg'].max(): 0.2f}\n Mean = ${df_low['Complics_reop_rate'].mean():0.6f} $"
    text2 = f"mid\n ${df_mid['total_year_avg'].min(): 0.2f} - ${df_mid['total_year_avg'].max(): 0.2f}\n Mean = ${df_mid['Complics_reop_rate'].mean():0.6f} $"
    text3 =f"high\n${df_high['total_year_avg'].min(): 0.2f} - ${df_high['total_year_avg'].max(): 0.2f}\n Mean = ${df_high['Complics_reop_rate'].mean():0.6f} $"
    labels = [text, text2, text3]


    fig1, ax1 = plt.subplots()
    ax1.set_title('Complication siteid reop boxplot')
    bp = ax1.boxplot(data, patch_artist=True, labels=labels)
    colors = ['pink', 'lightblue', 'palegreen']
    for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)

    f = lambda c: plt.plot([], color=c, ls="", marker="o")[0]
    plt.legend(handles=[f("pink"), f("lightblue"), f("palegreen")],
               labels=['low', 'mid', 'high'])
    plt.ylabel("Complication Reop rate")
    plt.show()

siteid_obs_vs_expected_mort()
siteid_obs_vs_expected_mort_reop()
surgid_obs_vs_expected_mort()
siteid_obs_vs_expected_complics()
surgid_obs_vs_expected_complics()

mortality_reop_surgid_boxplot()
mortality_reop_siteid_boxplot()
complics_reop_surgid_boxplot()
complics_reop_siteid_boxplot()


def mortality_reop_surgid_obs_vs_expec_boxplot():
    mask = df_mort_surgid['count_Reop'] == 0
    df_reop = df_mort_surgid[~mask]
    # total_year_sum
    new_df=pd.DataFrame(data=df_reop,columns=['log_Reoperation','total_year_avg'])

    new_df['bins'] = pd.qcut(new_df['total_year_avg'], 3, labels=['low', 'mid', 'high'])
    print(new_df)
    new_df.to_csv("box_surgid_reop_obs_ vs expected.csv")
    mask = new_df['bins'] == 'low'
    df_low = new_df[mask]
    mask = new_df['bins'] == 'mid'
    df_mid = new_df[mask]
    mask = new_df['bins'] == 'high'
    df_high = new_df[mask]

    data = [df_low['log_Reoperation'],df_mid['log_Reoperation'],df_high['log_Reoperation']]
    print (df_low.describe())
    print(df_mid.describe())
    print(df_high.describe())
    text = f" low\n ${df_low['total_year_avg'].min(): 0.2f} - ${df_low['total_year_avg'].max(): 0.2f}\n Mean = ${df_low['log_Reoperation'].mean():0.6f} $"
    text2 = f"mid\n ${df_mid['total_year_avg'].min(): 0.2f} - ${df_mid['total_year_avg'].max(): 0.2f}\n  Mean = ${df_mid['log_Reoperation'].mean():0.6f} $"
    text3 =f"high\n${df_high['total_year_avg'].min(): 0.2f} - ${df_high['total_year_avg'].max(): 0.2f}\n Mean = ${df_high['log_Reoperation'].mean():0.6f} $"
    # ax = plt.gca()
    # ax = sns.boxplot(x="day", y="total_bill", data=df_mid['mortalty_reop_rate'])
    # show plot
    labels = [text,text2,text3]
    #
    # ${df_low['log_Reoperation'].min(): 0.2f} - ${df_low['log_Reoperation'].max(): 0.2f}
    # ${df_mid['log_Reoperation'].min(): 0.2f} - ${df_mid['log_Reoperation'].max(): 0.2f}
    # ${df_high['log_Reoperation'].min(): 0.2f} - ${df_high['log_Reoperation'].max(): 0.2f}
    data = [df_low['log_Reoperation'],df_mid['log_Reoperation'],df_high['log_Reoperation']]
    r,p = scipy.stats.kruskal(df_low['total_year_avg'], df_low['log_Reoperation'])
    text = f" low:  p-value : {p:0.6f}"
    print(text)
    r, p = scipy.stats.kruskal(df_mid['total_year_avg'], df_mid['log_Reoperation'])
    text = f" mid:  p-value : {p:0.6f}"
    print(text)
    r, p = scipy.stats.kruskal(df_high['total_year_avg'], df_high['log_Reoperation'])
    text = f" high:  p-value : {p:0.6f}"
    print(text)
    fig1, ax1 = plt.subplots()
    ax1.set_title('mortality reop surgid observe vs expected ')
    bp = ax1.boxplot(data, patch_artist=True, labels=labels)
    colors = ['pink', 'lightblue', 'palegreen']
    for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)

    f = lambda c: plt.plot([], color=c, ls="", marker="o")[0]
    plt.legend(handles=[f("pink"), f("lightblue"), f("palegreen")],
               labels=['low', 'mid', 'high'])
    plt.ylabel("mortality reop surgid observe vs expected")
    plt.show()

def mortality_reop_siteid_obs_vs_expec_boxplot():
    mask = df_mort_siteid['count_Reop'] == 0
    df_reop = df_mort_siteid[~mask]
    # total_year_sum
    new_df=pd.DataFrame(data=df_reop,columns=['log_Reoperation','total_year_avg'])

    new_df['bins'] = pd.qcut(new_df['total_year_avg'], 3, labels=['low', 'mid', 'high'])
    print(new_df)
    new_df.to_csv("box_surgid_reop_obs_ vs expected.csv")
    mask = new_df['bins'] == 'low'
    df_low = new_df[mask]
    mask = new_df['bins'] == 'mid'
    df_mid = new_df[mask]
    mask = new_df['bins'] == 'high'
    df_high = new_df[mask]

    data = [df_low['log_Reoperation'],df_mid['log_Reoperation'],df_high['log_Reoperation']]
    r,p = scipy.stats.kruskal(df_low['total_year_avg'], df_low['log_Reoperation'])
    text = f" low:  p-value : {p:0.6f}"
    print(text)
    r, p = scipy.stats.kruskal(df_mid['total_year_avg'], df_mid['log_Reoperation'])
    text = f" mid:  p-value : {p:0.6f}"
    print(text)
    r, p = scipy.stats.kruskal(df_high['total_year_avg'], df_high['log_Reoperation'])
    text = f" high:  p-value : {p:0.6f}"
    print(text)


    text = f" low\n ${df_low['total_year_avg'].min(): 0.2f} - ${df_low['total_year_avg'].max(): 0.2f}\n Mean = ${df_low['log_Reoperation'].mean():0.6f} $"
    text2 = f"mid\n ${df_mid['total_year_avg'].min(): 0.2f} - ${df_mid['total_year_avg'].max(): 0.2f}\n  Mean = ${df_mid['log_Reoperation'].mean():0.6f} $"
    text3 = f"high\n${df_high['total_year_avg'].min(): 0.2f} - ${df_high['total_year_avg'].max(): 0.2f}\n Mean = ${df_high['log_Reoperation'].mean():0.6f} $"
    # ax = plt.gca()
    # ax = sns.boxplot(x="day", y="total_bill", data=df_mid['mortalty_reop_rate'])
    # show plot
    labels = [text,text2,text3]
    #
    # ${df_low['log_Reoperation'].min(): 0.2f} - ${df_low['log_Reoperation'].max(): 0.2f}
    # ${df_mid['log_Reoperation'].min(): 0.2f} - ${df_mid['log_Reoperation'].max(): 0.2f}
    # ${df_high['log_Reoperation'].min(): 0.2f} - ${df_high['log_Reoperation'].max(): 0.2f}

    fig1, ax1 = plt.subplots()
    ax1.set_title('mortality reop siteid observe vs expected ')
    bp = ax1.boxplot(data, patch_artist=True, labels=labels)
    colors = ['pink', 'lightblue', 'palegreen']
    for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)

    f = lambda c: plt.plot([], color=c, ls="", marker="o")[0]
    plt.legend(handles=[f("pink"), f("lightblue"), f("palegreen")],
               labels=['low', 'mid', 'high'])
    plt.ylabel("mortality reop siteid observe vs expected")
    plt.show()

mortality_reop_surgid_obs_vs_expec_boxplot()
mortality_reop_siteid_obs_vs_expec_boxplot()