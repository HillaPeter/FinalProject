import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats
from sklearn import linear_model
import statsmodels.api as sm
from scipy import stats
###################
yaara="723"
daniel = "957"
hilla="355"
generic_path = "/tmp/pycharm_project_"+hilla+"/"



#Full data
dfOp = pd.read_csv("/mnt/nadavrap-students/STS/data/Shapira_1st-Op_6_9_20_.csv")
groupOp = dfOp.groupby("SiteID")["SiteID"].count().reset_index(name='countFirst')
#draw a plot
x = groupOp["countFirst"]
plt.hist(x, bins=40)
plt.title("Histogram of count Operation")
plt.xlabel('number of Operations')
plt.ylabel('count of SiteId')
plt.show()
plt.savefig('Histogram of count Operation.png')

#ReOp data
dfReOp = pd.read_csv("/mnt/nadavrap-students/STS/data/Shapira_reOp_6_9_20_.csv")
groupReOp = dfReOp.groupby("SiteID")["SiteID"].count().reset_index(name='countReOp')
#draw a plot
y = groupReOp['countReOp']
plt.hist(y, bins=20)
plt.title("Histogram of count ReOperation")
plt.xlabel('number of ReOperations')
plt.ylabel('count of SiteId')
plt.show()
plt.savefig('Histogram of count ReOperation.png')

##merge two dataframes into one and gets the ratio between them
result = pd.merge(groupOp, groupReOp, on='SiteID', how='left')
result['countReOp'].fillna(0, inplace=True)
result["countReOp/countFirst+countReOp"] =  (result["countReOp"] /(result["countReOp"]+ result["countFirst"])) *100
result['countReOp/countFirst+countReOp'].fillna(0, inplace=True)
result.to_csv(generic_path+"result.csv")
#draw a plot
z = result['countReOp/countFirst+countReOp']
plt.hist(z, bins=40)
plt.title("Histogram of  ReOperation vs Operation")
plt.xlabel('% of ReOperation of Operation')
plt.ylabel('count of SiteId')
plt.show()
plt.savefig('Histogram of ReOperation vs Operation.png')

########### nadav recomend ###############
# import feather
# feather.write_dataframe(dfOp, "/tmp/pycharm_project_723/dfOp.feather")
# feather.write_dataframe(dfReOp, "/tmp/pycharm_project_723/dfReOp.feather")
# dfop1 = feather.read_dataframe("/tmp/pycharm_project_723/dfOp.feather")
# dfReOp1 = feather.read_dataframe("/tmp/pycharm_project_723/dfReOp.feather")

######mortality
MortaltyOp = dfOp.groupby('SiteID')['Mortalty'].apply(lambda x: (x== 1 ).sum()).reset_index(name='Mortalty_SiteID_op')
MortaltyReOp = dfReOp.groupby('SiteID')['Mortalty'].apply(lambda x: (x== 1 ).sum()).reset_index(name='Mortalty_SiteID_reOp')
result2 = pd.merge(MortaltyOp, MortaltyReOp, on='SiteID', how='left')
# result.merge(result2, on='SiteID')
df=pd.merge(result, result2, on='SiteID')
df["countOpr"] = result["countReOp"]+ result["countFirst"]
df.to_csv(generic_path+"mortalty.csv")




####AGE

ageOp = dfOp.groupby("SiteID")["Age"].mean().reset_index(name='Mean_Age_op')
ageReOp = dfReOp.groupby("SiteID")["Age"].mean().reset_index(name='Mean_Age_reOp')
resultAge = pd.merge(ageOp, ageReOp, on='SiteID', how='left')
dfAge=pd.merge(result, resultAge, on='SiteID')


genderOp = pd.get_dummies(dfOp["Gender"]).rename(columns=lambda x: 'opGender_' + str(x))
dfOp=dfOp.join(genderOp)
genderReOp = pd.get_dummies(dfReOp["Gender"]).rename(columns=lambda x: 'reOpGender_' + str(x))
dfReOp=dfReOp.join(genderReOp)
genderOp_grouped_male = (dfOp.groupby("SiteID")["opGender_1.0"]).sum().reset_index(name='male_Op')
genderOp_grouped_female = (dfOp.groupby("SiteID")["opGender_2.0"]).sum().reset_index(name='female_Op')
dfMale=pd.merge(genderOp_grouped_male, genderOp_grouped_female, on='SiteID')
genderReOp_grouped_male = (dfReOp.groupby("SiteID")["reOpGender_1.0"]).sum().reset_index(name='male_reOp')
genderReOp_grouped_female = (dfReOp.groupby("SiteID")["reOpGender_2.0"]).sum().reset_index(name='female_reOp')
dfFemale=pd.merge(genderReOp_grouped_male, genderReOp_grouped_female, on='SiteID')
dfGender=pd.merge(dfMale, dfFemale, on='SiteID')

dfMerge=pd.merge(dfAge,dfGender, on='SiteID')


##FHCAD - family history of disease
FHCADOp = dfOp.groupby('SiteID')['FHCAD'].apply(lambda x: (x== 1 ).sum()).reset_index(name='FHCAD_op')
FHCADReOp =dfReOp.groupby('SiteID')['FHCAD'].apply(lambda x: (x== 1 ).sum()).reset_index(name='FHCAD_reOp')
resultFHCAD = pd.merge(FHCADOp, FHCADReOp, on='SiteID', how='left')
dfFHCAD =pd.merge(dfMerge, resultFHCAD, on='SiteID')


##Hypertn - blood preasure
HypertnOp = dfOp.groupby("SiteID")["Hypertn"].apply(lambda x: (x== 1 ).sum()).reset_index(name='Hypertn_op')
HypertnReOp = dfReOp.groupby("SiteID")["Hypertn"].apply(lambda x: (x== 1 ).sum()).reset_index(name='Hypertn_reOp')
resultHypertn = pd.merge(HypertnOp, HypertnReOp, on='SiteID', how='left')
dfHypertn =pd.merge(dfFHCAD, resultHypertn, on='SiteID')

##Diabetes
DiabetesOp =  dfOp.groupby('SiteID')['Diabetes'].apply(lambda x: (x== 1 ).sum()).reset_index(name='Diabetes_op')
DiabetesReOp = dfReOp.groupby('SiteID')['Diabetes'].apply(lambda x: (x== 1 ).sum()).reset_index(name='Diabetes_reOp')
resultDiabetes = pd.merge(DiabetesOp, DiabetesReOp, on='SiteID', how='left')
dfDiabetes =pd.merge(dfHypertn, resultDiabetes, on='SiteID')

##Dyslip
DyslipOp = dfOp.groupby("SiteID")["Dyslip"].apply(lambda x: (x== 1 ).sum()).reset_index(name='Dyslip_op')
DyslipReOp = dfReOp.groupby("SiteID")["Dyslip"].apply(lambda x: (x== 1 ).sum()).reset_index(name='Dyslip_reOp')
resultDyslip = pd.merge(DyslipOp, DyslipReOp, on='SiteID', how='left')
dfDyslip =pd.merge(dfDiabetes, resultDyslip, on='SiteID')

##TobaccoUse
smokeEveryDayOp = dfOp.groupby("SiteID")["TobaccoUse"].apply(lambda x: ((x>= 2) & (x<6) ).sum()).reset_index(name='smoke_op')
smokeEveryDayReOp = dfReOp.groupby("SiteID")["TobaccoUse"].apply(lambda x: ((x>= 2) & (x<6) ).sum()).reset_index(name='smoke_reOp')
resultSmoke = pd.merge(smokeEveryDayOp, smokeEveryDayReOp, on='SiteID', how='left')
dfTobaccoUseResult =pd.merge(dfDyslip, resultSmoke, on='SiteID')

##Cancer
CancerOp = dfOp.groupby("SiteID")["Cancer"].apply(lambda x: (x== 1 ).sum()).reset_index(name='Cancer_op')
CancerReOp = dfReOp.groupby("SiteID")["Cancer"].apply(lambda x: (x== 1 ).sum()).reset_index(name='Cancer_reOp')
resultCancer = pd.merge(CancerOp, CancerReOp, on='SiteID', how='left')
dfCancer =pd.merge(dfTobaccoUseResult, resultCancer, on='SiteID')

##PVD
PVDOp = dfOp.groupby("SiteID")["PVD"].apply(lambda x: (x== 1 ).sum()).reset_index(name='PVD_op')
PVDReOp = dfReOp.groupby("SiteID")["PVD"].apply(lambda x: (x== 1 ).sum()).reset_index(name='PVD_reOp')
resultPVD = pd.merge(PVDOp, PVDReOp, on='SiteID', how='left')
dfPVD =pd.merge(dfCancer, resultPVD, on='SiteID')

dfPVD.to_csv(generic_path+"riskFactors.csv")
# df=pd.read_csv("mortalty.csv")
#reOp

df['mortalPerReOp']=(df['Mortalty_SiteID_reOp']/df['countReOp'])*100
df['prop']=df['countReOp/countFirst+countReOp']

#1
df.plot(kind='scatter', x='countOpr', y='mortalPerReOp', title="Mortality of reOp - total Ops")
plt.show()
plt.savefig('Mortality of reOp - total Ops.png')

#2
df.plot(kind='scatter', x='countReOp', y='mortalPerReOp', title="Mortality of reOp - reOps")
plt.show()
plt.savefig('Mortality of reOp - reOps.png')

#3
df.plot(kind='scatter', x='countReOp/countFirst+countReOp', y='mortalPerReOp', title="Mortality of reOp - reOp/(reOp+Ops)")
plt.show()
plt.savefig('Mortality of reOp - reOp_reOp+Ops.png')

###oP
#1
df['mortalPerOp']=(df['Mortalty_SiteID_op']/df['countFirst'])*100
df.plot(kind='scatter', x='countOpr', y='mortalPerOp', title="Mortality of op - total Ops")
plt.show()
plt.savefig('Mortality of op - total Ops.png')

#2
df.plot(kind='scatter', x='countFirst', y='mortalPerOp', title="Mortality of op - ops")
plt.show()
plt.savefig('Mortality of op - ops.png')

#3
df.plot(kind='scatter', x='countReOp/countFirst+countReOp', y='mortalPerOp', title="Mortality of op - reOp/(reOp+Ops)")
plt.show()
plt.savefig('Mortality of op - reOp_reOp+Ops.png')

df_mortality=pd.read_csv("mortalty.csv")
df_mortality['Mortalty_SiteID_reOp']=df_mortality['Mortalty_SiteID_reOp'].fillna(0)
df_mortality['prop']=df_mortality['countReOp']/df_mortality['countOpr']

#spearman
print("spearman")
print("Reop:")
print("pvalue ReOp all ops(include reOps) ",stats.spearmanr(a=df_mortality['Mortalty_SiteID_reOp'],b=df_mortality['countOpr'],nan_policy='omit')[1])
print("pvalue ReOp only reOps ",stats.spearmanr(a=df_mortality['countFirst'],b=df_mortality['countOpr'],nan_policy='omit')[1])
print("pvalue ReOp prop re/total " ,stats.spearmanr(a=df_mortality['Mortalty_SiteID_reOp'],b=df_mortality['prop'],nan_policy='omit')[1])
print()
#op
print("First op:")
print("pvalue FirstOp all ops(include reOps) ",stats.spearmanr(a=df_mortality['Mortalty_SiteID_op'],b=df_mortality['countOpr'],nan_policy='omit')[1])
print("pvalue FirstOp only Ops ",stats.spearmanr(a=df_mortality['Mortalty_SiteID_op'],b=df_mortality['countFirst'],nan_policy='omit')[1])
print("pvalue FirstOp prop re/total " , stats.spearmanr(a=df_mortality['Mortalty_SiteID_op'],b=df_mortality['prop'],nan_policy='omit')[1])
print("")

#pearson
print("pearson")
#reOp
print("Reop:")
print("pvalue ReOp all ops(include reOps) ",stats.pearsonr(x=df_mortality['Mortalty_SiteID_reOp'],y=df_mortality['countOpr'])[1])
print("pvalue ReOp only reOps ",stats.pearsonr(df_mortality['Mortalty_SiteID_reOp'],df_mortality['countFirst'])[1])
print("pvalue ReOp prop re/total " ,stats.pearsonr(df_mortality['Mortalty_SiteID_reOp'],df_mortality['prop'])[1])

#op
print("First op:")
print("pvalue FirstOp all ops(include reOps) ",stats.pearsonr(x=df_mortality['Mortalty_SiteID_op'],y=df_mortality['countOpr'])[1])
print("pvalue FirstOp only reOps ",stats.pearsonr(df_mortality['Mortalty_SiteID_op'],df_mortality['countFirst'])[1])
print("pvalue FirstOp prop re/total ",stats.pearsonr(df_mortality['Mortalty_SiteID_op'],df_mortality['prop'])[1])



df_risk=pd.read_csv("riskFactors.csv")


df= pd.merge(df_risk,df_mortality, on='SiteID')

# df.drop(columns=['Unnamed: 0_x'])
del df['Unnamed: 0_x']
del df['Unnamed: 0_y']
df.to_csv("merge.csv", index=False)


perOp = {
    'Mortalty_SiteID_op': (df['Mortalty_SiteID_op']/df['countFirst_x']) * 100,
    'SiteID': df['SiteID'],
    'Mean_Age_op': df['Mean_Age_op'],
    'female_Op': (df['female_Op']/df['countFirst_x']) * 100,
    'male_Op': (df['male_Op']/df['countFirst_x']) * 100,
    'FHCAD_op': (df['FHCAD_op']/df['countFirst_x']) * 100,
    'Hypertn_op': (df['Hypertn_op']/df['countFirst_x']) * 100,
    'Diabetes_op': (df['Diabetes_op']/df['countFirst_x']) * 100,
    'Dyslip_op': (df['Dyslip_op']/df['countFirst_x']) * 100,
    'smoke_op': (df['smoke_op']/df['countFirst_x']) * 100,
    'Cancer_op': (df['Cancer_op']/df['countFirst_x']) * 100,
    'PVD_op': (df['PVD_op']/df['countFirst_x']) * 100,
    }

dfOp = pd.DataFrame(perOp, columns=['SiteID','Mortalty_SiteID_op','Mean_Age_op', 'female_Op','male_Op','FHCAD_op','Hypertn_op','Diabetes_op','Dyslip_op','smoke_op','Cancer_op','PVD_op'])
dfOp.to_csv(generic_path+"opPercentage.csv", index=False)


perReOp = {
    'Mortalty_SiteID_reOp': (df['Mortalty_SiteID_reOp']/df['countOpr']) * 100,
    'SiteID': df['SiteID'],
    'Mean_Age_reOp':df['Mean_Age_reOp'],
    'female_reOp': (df['female_reOp']/df['countOpr']) * 100,
    'male_reOp': (df['male_reOp']/df['countOpr']) * 100,
    'FHCAD_reOp': (df['FHCAD_reOp']/df['countOpr']) * 100,
    'Hypertn_reOp': (df['Hypertn_reOp']/df['countOpr']) * 100,
    'Diabetes_reOp': (df['Diabetes_reOp']/df['countOpr']) * 100,
    'Dyslip_reOp': (df['Dyslip_reOp']/df['countOpr']) * 100,
    'smoke_reOp': (df['smoke_reOp']/df['countOpr']) * 100,
    'Cancer_reOp': (df['Cancer_reOp']/df['countOpr']) * 100,
    'PVD_reOp': (df['PVD_reOp']/df['countOpr']) * 100,
    }


dfReOp = pd.DataFrame(perReOp, columns = ['SiteID','Mortalty_SiteID_reOp','Mean_Age_reOp', 'female_reOp','male_reOp','FHCAD_reOp','Hypertn_reOp','Diabetes_reOp','Dyslip_reOp','smoke_reOp','Cancer_reOp','PVD_reOp'])
dfReOp.to_csv("reOpPercentage.csv", index=False)

#op!
print()
print("-------Op-------")
X = dfOp[['Mean_Age_op', 'female_Op','male_Op','FHCAD_op','Hypertn_op','Diabetes_op','Dyslip_op','smoke_op','Cancer_op','PVD_op']]  # here we have 2 variables for multiple regression. If you just want to use one variable for simple linear regression, then use X = df['Interest_Rate'] for example.Alternatively, you may add additional variables within the brackets
Y = dfOp['Mortalty_SiteID_op']

# with sklearn
regr = linear_model.LinearRegression()
regr.fit(X, Y)

print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)

# # prediction with sklearn
# New_Interest_Rate = 2.75
# New_Unemployment_Rate = 5.3
# print('Predicted Stock Index Price: \n', regr.predict([[New_Interest_Rate, New_Unemployment_Rate]]))

# with statsmodels
X = sm.add_constant(X)  # adding a constant

model = sm.OLS(Y, X).fit()
predictions = model.predict(X)

print_model = model.summary()
print(print_model)

print()
print("-------ReOp-------")
X1 = dfReOp[['Mean_Age_reOp', 'female_reOp','male_reOp','FHCAD_reOp','Hypertn_reOp','Diabetes_reOp','Dyslip_reOp','smoke_reOp','Cancer_reOp','PVD_reOp']]  # here we have 2 variables for multiple regression. If you just want to use one variable for simple linear regression, then use X = df['Interest_Rate'] for example.Alternatively, you may add additional variables within the brackets
Y1 = dfReOp['Mortalty_SiteID_reOp']

# with sklearn
regr1 = linear_model.LinearRegression()
regr1.fit(X1, Y1)

print('Intercept: \n', regr1.intercept_)
print('Coefficients: \n', regr1.coef_)

# # prediction with sklearn
# New_Interest_Rate = 2.75
# New_Unemployment_Rate = 5.3
# print('Predicted Stock Index Price: \n', regr.predict([[New_Interest_Rate, New_Unemployment_Rate]]))

# with statsmodels
X1 = sm.add_constant(X1)  # adding a constant

model1 = sm.OLS(Y1, X1).fit()
predictions1 = model1.predict(X1)

print_model1 = model1.summary()
print(print_model1)