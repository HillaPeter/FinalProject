import pandas as pd


df = pd.DataFrame({'Foo': ['A','B','C','D','E'],
'Score1': [4,6,2,7,8]
})

df.columns=['Foo','Score1']

df2 = pd.DataFrame({'Foo': ['A','C','D','E'],
'Score2': [5,10,10,5]
})
df2.columns=['Foo','Score2']
result = pd.merge(df, df2, on='Foo', how='left')
print(result)
result=result.dropna()

result["Score1/Score2"] = result["Score1"] / result["Score2"]

# result = result.iloc[1:]

print(result)