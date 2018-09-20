import pandas as pd
from sklearn.decomposition import PCA
df = pd.read_csv('../data/creditcard.csv')

df = df.reset_index()
df = df.dropna()
df.info()

#make a PCA object that retains 95% of the variance of the dataset
pca = PCA(0.95)

#apply PCA to df
pca.fit(df)
df = pca.transform(df)
df.info()

