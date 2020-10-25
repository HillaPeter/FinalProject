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



