from sklearn.ensemble.iforest import IsolationForest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df_wine = pd.read_csv('../data/wine_data.txt', header=None)
df_wine.columns = ['Category_label','Alcohol','Malic acid','Ash','Alcalinity of ash',
                   'Magnesium','Total phenols','Flavanoids','Nonflavanoid phenols',
                   'Proanthocyanins','Color intensity','Hue','OD280/OD315 of diluted wines',
                   'Proline']

df_wine = df_wine[(df_wine['Category_label'] == 1) | (df_wine['Category_label'] == 2)]
temp = df_wine.drop(axis=1,columns=['Category_label'])


print(df_wine.info())
print(temp.info())

model = IsolationForest(n_estimators=df_wine.size, random_state=2018,
                        contamination=min(df_wine['Category_label'].value_counts())/temp.size)
model.fit(temp)
scores = model.decision_function(temp)

# Note that the internal treshold is set to
# expected contamination percentile of the scores.
# if contamination=0.1. Than lowest 10% of scores are considered outliers.
inliers = model.predict(temp)

plt.figure()
plt.title('Scores wine dataset')
plt.hist(scores[inliers==1], bins=100, label='wine category 1')
plt.hist(scores[inliers==-1], bins=50, label='wine category 2')
plt.legend(loc='top right')
plt.show()

index = np.arange(inliers.size)
cond_1 = df_wine['Category_label']==1

plt.figure()
plt.title('Predicted vs actual inliers wine dataset')
plt.scatter(index[inliers==1], inliers[inliers==1],color='g', marker='o',label='predicted inlier',alpha=0.2)
plt.scatter(index[~inliers==1], inliers[~inliers==1],color='r', marker='o',label='predicted outlier',alpha=0.2)
plt.scatter(index[cond_1], df_wine['Category_label'][cond_1], color='g',marker='x', label='actual inlier',alpha=0.)
plt.scatter(index[~cond_1], df_wine['Category_label'][~cond_1], color='r',marker='x', label='actual outlier',alpha=0.2)
plt.legend(loc= 'top_right')
plt.show()