# #import modin.pandas as pd
# import pandas as pd
# import matplotlib.pyplot as plt
# import asyncio
#
#
# ###################
# async def readData(path, nameIndex):
#   data = await pd.read_csv(path)
#   groupData = data.groupby("SiteID")["SiteID"].count().reset_index(name=nameIndex)
#   return groupData
#
# async def main():
#   values = await asyncio.gather(readData("/mnt/nadavrap-students/STS/data/Shapira_1st-Op_6_9_20_.csv", 'countOp'), readData("/mnt/nadavrap-students/STS/data/Shapira_reOp_6_9_20_.csv", 'countReOp'))
#   groupOp = values[0]
#   groupReOp = values[1]
#   #Full data
#   #groupOp = readData("/mnt/nadavrap-students/STS/data/Shapira_1st-Op_6_9_20_.csv", 'countOp')
#   #ReOp data
#   #groupReOp = readData("/mnt/nadavrap-students/STS/data/Shapira_reOp_6_9_20_.csv", 'countReOp')
#
#   ##merge two dataframes into one and gets the ratio between them
#   result = pd.merge(groupOp, groupReOp, on='SiteID', how='left')
#   result=result.dropna()
#   result["countReOp/countOp"] =  (result["countReOp"] / result["countOp"]) *100
#
#   #draw a plot
#   ax = result.plot.bar(x='SiteID', y='countReOp/countOp', rot=0)
#   plt.show()
#
# asyncio.run(main())


import pandas as pd
import matplotlib.pyplot as plt

###################

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

##merge two dataframes into one and gets the ratio between them
result = pd.merge(groupOp, groupReOp, on='SiteID', how='left')
result['countReOp'].fillna(0, inplace=True)
result["countReOp/countFirst+countReOp"] =  (result["countReOp"] /(result["countReOp"]+ result["countFirst"])) *100
result['countReOp/countFirst+countReOp'].fillna(0, inplace=True)
# result.to_csv("/tmp/pycharm_project_355/result.csv")
#draw a plot
z = result['countReOp/countFirst+countReOp']
plt.hist(z, bins=40)
plt.title("Histogram of  ReOperation vs Operation")
plt.xlabel('% of ReOperation of Operation')
plt.ylabel('count of SiteId')
plt.show()

########### nadav recomend ###############
# import feather
# feather.write_dataframe(dfOp, "/tmp/pycharm_project_723/dfOp.feather")
# feather.write_dataframe(dfReOp, "/tmp/pycharm_project_723/dfReOp.feather")
# dfop1 = feather.read_dataframe("/tmp/pycharm_project_723/dfOp.feather")
# dfReOp1 = feather.read_dataframe("/tmp/pycharm_project_723/dfReOp.feather")

######mortality
MortaltyOp = dfOp.groupby("SiteID")["Mortalty"].count().reset_index(name='Mortalty_SiteID_op')
MortaltyReOp = dfReOp.groupby("SiteID")["Mortalty"].count().reset_index(name='Mortalty_SiteID_reOp')
result2 = pd.merge(MortaltyOp, MortaltyReOp, on='SiteID', how='left')
# result.merge(result2, on='SiteID')
df=pd.merge(result, result2, on='SiteID')
df["countOpr"] = result["countReOp"]+ result["countFirst"]
df.to_csv("/tmp/pycharm_project_355/mortalty.csv")




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
FHCADOp = dfOp.groupby("SiteID")["FHCAD"].count().reset_index(name='FHCAD_op')
FHCADReOp = dfReOp.groupby("SiteID")["FHCAD"].count().reset_index(name='FHCAD_reOp')
resultFHCAD = pd.merge(FHCADOp, FHCADReOp, on='SiteID', how='left')
dfFHCAD =pd.merge(dfMerge, resultFHCAD, on='SiteID')

##Hypertn - blood preasure
HypertnOp = dfOp.groupby("SiteID")["Hypertn"].count().reset_index(name='Hypertn_op')
HypertnReOp = dfReOp.groupby("SiteID")["Hypertn"].count().reset_index(name='Hypertn_reOp')
resultHypertn = pd.merge(HypertnOp, HypertnReOp, on='SiteID', how='left')
dfHypertn =pd.merge(dfFHCAD, resultHypertn, on='SiteID')

##Diabetes
DiabetesOp = dfOp.groupby("SiteID")["Diabetes"].count().reset_index(name='Diabetes_op')
DiabetesReOp = dfReOp.groupby("SiteID")["Diabetes"].count().reset_index(name='Diabetes_reOp')
resultDiabetes = pd.merge(DiabetesOp, DiabetesReOp, on='SiteID', how='left')
dfDiabetes =pd.merge(dfHypertn, resultDiabetes, on='SiteID')

##Dyslip
DyslipOp = dfOp.groupby("SiteID")["Dyslip"].count().reset_index(name='Dyslip_op')
DyslipReOp = dfReOp.groupby("SiteID")["Dyslip"].count().reset_index(name='Dyslip_reOp')
resultDyslip = pd.merge(DyslipOp, DyslipReOp, on='SiteID', how='left')
dfDyslip =pd.merge(dfDiabetes, resultDyslip, on='SiteID')

##TobaccoUse
TobaccoUseOp = dfOp.groupby("SiteID")["TobaccoUse"].count().reset_index(name='TobaccoUse_op')
TobaccoUseReOp = dfReOp.groupby("SiteID")["TobaccoUse"].count().reset_index(name='TobaccoUse_reOp')
resultTobaccoUse = pd.merge(TobaccoUseOp, TobaccoUseReOp, on='SiteID', how='left')
dfTobaccoUse =pd.merge(dfDyslip, resultTobaccoUse, on='SiteID')

##Cancer
CancerOp = dfOp.groupby("SiteID")["Cancer"].count().reset_index(name='Cancer_op')
CancerReOp = dfReOp.groupby("SiteID")["Cancer"].count().reset_index(name='Cancer_reOp')
resultCancer = pd.merge(CancerOp, CancerReOp, on='SiteID', how='left')
dfCancer =pd.merge(dfTobaccoUse, resultCancer, on='SiteID')

##PVD
PVDOp = dfOp.groupby("SiteID")["PVD"].count().reset_index(name='PVD_op')
PVDReOp = dfReOp.groupby("SiteID")["PVD"].count().reset_index(name='PVD_reOp')
resultPVD = pd.merge(PVDOp, PVDReOp, on='SiteID', how='left')
dfPVD =pd.merge(dfCancer, resultPVD, on='SiteID')

dfPVD.to_csv("/tmp/pycharm_project_355/riskFactors.csv")
# df=pd.read_csv("mortalty.csv")
#reOp

df['mortalPerReOp']=(df['Mortalty_SiteID_reOp']/df['countReOp'])*100
df['prop']=df['countReOp/countFirst+countReOp']

#1
df.plot(kind='scatter', x='countOpr', y='mortalPerReOp', title="Mortality of reOp - total Ops")
plt.show()

#2
df.plot(kind='scatter', x='countReOp', y='mortalPerReOp', title="Mortality of reOp - reOps")
plt.show()

#3
df.plot(kind='scatter', x='countReOp/countFirst+countReOp', y='mortalPerReOp', title="Mortality of reOp - reOp/(reOp+Ops)")
plt.show()

###oP
#1
df['mortalPerOp']=(df['Mortalty_SiteID_op']/df['countFirst'])*100
df.plot(kind='scatter', x='countOpr', y='mortalPerOp', title="Mortality of op - total Ops")
plt.show()

#2
df.plot(kind='scatter', x='countFirst', y='mortalPerOp', title="Mortality of op - ops")
plt.show()

#3
df.plot(kind='scatter', x='countReOp/countFirst+countReOp', y='mortalPerOp', title="Mortality of op - reOp/(reOp+Ops)")
plt.show()


#spearman
print("spearman")
#reOp
print(df.mortalPerReOp.corr(df.countOpr, method="spearman"))
print(df.mortalPerReOp.corr(df.countReOp, method="spearman"))
print(df.mortalPerReOp.corr(df.prop, method="spearman"))

#op
print(df.mortalPerOp.corr(df.countOpr, method="spearman"))
print(df.mortalPerOp.corr(df.countFirst, method="spearman"))
print(df.mortalPerOp.corr(df.prop, method="spearman"))


#pearson
print("pearson")
#reOp
print(df.mortalPerReOp.corr(df.countOpr, method="pearson"))
print(df.mortalPerReOp.corr(df.countReOp, method="pearson"))
print(df.mortalPerReOp.corr(df.prop, method="pearson"))

#op
print(df.mortalPerOp.corr(df.countOpr, method="pearson"))
print(df.mortalPerOp.corr(df.countFirst, method="pearson"))
print(df.mortalPerOp.corr(df.prop, method="pearson"))


