import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
import statsmodels.api as sm

path="/tmp/pycharm_project_355/"

df_siteid= pd.read_csv(path+"siteid_new_data.csv")
df_surgid= pd.read_csv(path+"surgid_new_data.csv")
df_hospid= pd.read_csv(path+"hospid_new_data.csv")



def mortality_total(df,title):
    df.plot(kind='scatter', x='total', y='mt30stat', title=title)
    plt.show()

# mortality_total(df_siteid, " site id: mortality - total ops")
# mortality_total(df_surgid, " surgeon id: mortality - total ops")
# mortality_total(df_hospid, " hospital id: mortality - total ops")



def draw_hist(data,num_of_bins,title,x_title,y_title):
    plt.hist(data, bins=num_of_bins)
    plt.title(title)
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.show()

# draw_hist(df_siteid['precent_reop/total'],20,"siteid Histogram of  ReOperation vs Operation",'% of ReOperation of Operation',"count of siteid")
# draw_hist(df_surgid['precent_reop/total'],20,"surgid Histogram of  ReOperation vs Operation",'% of ReOperation of Operation',"count of surgid")
# draw_hist(df_hospid['precent_reop/total'],20,"hospid Histogram of  ReOperation vs Operation",'% of ReOperation of Operation',"count of hospid")
#

def create_df_for_regression(df,title,df_title):
    new_df = {
    'mt30stat': df['precent_mortality/total'],
    df_title: df[df_title],
    'mean_age': df['mean_age'],
    'mean_weightkg': df['mean_weightkg'],
    'female': (df['female']/df['total']) * 100,
    'male': (df['male']/df['total']) * 100,
    'diabetes': (df['diabetes']/df['total']) * 100,
    'fhcad': (df['fhcad']/df['total']) * 100,
    'mean_predmort': df['mean_predmort'],
    'tobaccouse': (df['tobaccouse'] / df['total']) * 100,
    'chrlungd': (df['chrlungd'] / df['total']) * 100,
    'cancer': (df['cancer'] / df['total']) * 100,
    'pvd': (df['pvd'] / df['total']) * 100,
    'liverdis': (df['liverdis'] / df['total']) * 100,
    'alcohol': (df['alcohol'] / df['total']) * 100,
    }

    df_new_reg = pd.DataFrame(new_df, columns=[df_title,'mt30stat','mean_age', 'mean_weightkg','female','male','diabetes','fhcad','mean_predmort','tobaccouse','chrlungd','cancer','pvd','liverdis','alcohol'])
    df_new_reg.to_csv(title, index=False)

# create_df_for_regression(df_siteid,path+"siteid_regression.csv","siteid")
# create_df_for_regression(df_surgid,path+"surgid_regression.csv","surgid")
# create_df_for_regression(df_hospid,path+"hospid_regression.csv","hospid")


def create_regression(df,df_title):
    X = df[['mt30stat','mean_age', 'mean_weightkg','female','male','diabetes','fhcad','mean_predmort','tobaccouse','chrlungd','cancer','pvd','liverdis','alcohol']]  # here we have 2 variables for multiple regression. If you just want to use one variable for simple linear regression, then use X = df['Interest_Rate'] for example.Alternatively, you may add additional variables within the brackets
    Y = df[df_title]

    # with sklearn
    regr = linear_model.LinearRegression()
    regr.fit(X, Y)
    print('Intercept: \n', regr.intercept_)
    print('Coefficients: \n', regr.coef_)
    X = sm.add_constant(X)  # adding a constant
    model = sm.OLS(Y, X).fit()
    predictions = model.predict(X)
    print_model = model.summary()
    print(print_model)

create_regression(df_siteid,"siteid")
create_regression(df_surgid,"surgid")
create_regression(df_hospid,"hospid")