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
groupOp = dfOp.groupby("SiteID")["SiteID"].count().reset_index(name='countOp')
#draw a plot
x = groupOp["countOp"]
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
result["countReOp/countOp"] =  (result["countReOp"] /(result["countReOp"]+ result["countOp"])) *100
result['countReOp/countOp'].fillna(0, inplace=True)
result.to_csv("/tmp/pycharm_project_723/result.csv")
#draw a plot
z = result['countReOp/countOp']
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
df["totalOp"] = result["countReOp"]+ result["countOp"]
df.to_csv("/tmp/pycharm_project_355/mortalty.csv")



# df=pd.read_csv("mortalty.csv")
#reOp

df['mortalPerReOp']=(df['Mortalty_SiteID_reOp']/df['countReOp'])*100
df['prop']=df['countReOp/countOp']

#1
df.plot(kind='scatter', x='totalOp', y='mortalPerReOp', title="Mortality of reOp - total Ops")
plt.show()

#2
df.plot(kind='scatter', x='countReOp', y='mortalPerReOp', title="Mortality of reOp - reOps")
plt.show()

#3
df.plot(kind='scatter', x='countReOp/countOp', y='mortalPerReOp', title="Mortality of reOp - reOp/(reOp+Ops)")
plt.show()

###oP
#1
df['mortalPerOp']=(df['Mortalty_SiteID_op']/df['countOp'])*100
df.plot(kind='scatter', x='totalOp', y='mortalPerOp', title="Mortality of op - total Ops")
plt.show()

#2
df.plot(kind='scatter', x='countOp', y='mortalPerOp', title="Mortality of op - ops")
plt.show()

#3
df.plot(kind='scatter', x='countReOp/countOp', y='mortalPerOp', title="Mortality of op - reOp/(reOp+Ops)")
plt.show()


#spearman
print("spearman")
#reOp
print(df.mortalPerReOp.corr(df.totalOp, method="spearman"))
print(df.mortalPerReOp.corr(df.countReOp, method="spearman"))
print(df.mortalPerReOp.corr(df.prop, method="spearman"))

#op
print(df.mortalPerOp.corr(df.totalOp, method="spearman"))
print(df.mortalPerOp.corr(df.countOp, method="spearman"))
print(df.mortalPerOp.corr(df.prop, method="spearman"))


#pearson
print("pearson")
#reOp
print(df.mortalPerReOp.corr(df.totalOp, method="pearson"))
print(df.mortalPerReOp.corr(df.countReOp, method="pearson"))
print(df.mortalPerReOp.corr(df.prop, method="pearson"))

#op
print(df.mortalPerOp.corr(df.totalOp, method="pearson"))
print(df.mortalPerOp.corr(df.countOp, method="pearson"))
print(df.mortalPerOp.corr(df.prop, method="pearson"))


