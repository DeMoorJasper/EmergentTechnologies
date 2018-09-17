# This is an example of IsolationForest on the Wine data set
# Anomalies are not few and different. In fact they are frequent and similar!
# This doesn't produce satisfactory results.

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

model = IsolationForest(max_samples=temp.shape[0]//5, n_estimators=temp.shape[0],
                        random_state=2018, contamination=min(df_wine['Category_label'].value_counts())/temp.shape[0])
model.fit(temp)
scores = model.decision_function(temp)

# Note that the internal treshold is set to
# expected contamination percentile of the scores.
# if contamination=0.1. Than lowest 10% of scores are considered outliers.
inliers = model.predict(temp)

plt.figure()
plt.title('Scores wine dataset')
plt.hist(scores[inliers==1], bins=100, label='wine category 1')
plt.hist(scores[inliers==-1], bins=100, label='wine category 2')
plt.legend()
plt.show()

index = np.arange(inliers.size)
cond_pred = inliers==1
cond_actual = np.array(df_wine['Category_label']==1)
ones = np.ones(inliers.size)
ones_neg = -1*ones

plt.figure()
plt.title('Predicted vs actual inliers wine dataset')
plt.scatter(index[cond_pred], inliers[cond_pred],color='g', marker='o',label='predicted inlier',alpha=0.2)
plt.scatter(index[~cond_pred], inliers[~cond_pred],color='r', marker='o',label='predicted outlier',alpha=0.2)
plt.plot(index[cond_actual], ones[cond_actual], color='b', label='actual inlier')
plt.plot(index[~cond_actual], ones_neg[~cond_actual], color='b', label='actual outlier')
plt.legend()
plt.ylim([-2, 2])
plt.show()