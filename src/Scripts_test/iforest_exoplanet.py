# This is an example of IsolationForest on the Exoplanet data set
# Anomalies are few but not different enough for vanilla Iforests.
# This doesn't produce satisfactory results.
# better to take a very simple dataset that can be divided in inliers/outliers
# same for the exercices.

from sklearn.ensemble.iforest import IsolationForest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('../data/exoTest.csv')

df = df.dropna()
df = df.reset_index()

temp = df.drop(axis=1,columns=['index','LABEL'])

print(df.info())
print(temp.info())


# Note that the internal treshold is set to
# expected contamination percentile of the scores.
# if contamination=0.1. Than lowest 10% of scores are considered outliers.
model = IsolationForest(max_samples=temp.shape[0], n_estimators=temp.shape[0],
                        random_state=2018, contamination=min(df['LABEL'].value_counts())/temp.shape[0])
model.fit(temp)
scores = model.decision_function(temp)

inliers = model.predict(temp)
Label_count = df['LABEL'].value_counts()
plt.figure()
plt.title('Scores Exo planet dataset')
plt.hist(scores[inliers==1], bins=Label_count[1], label='non-exo-planet')
plt.hist(scores[inliers==-1], bins=Label_count[2], label='exo-planet')
plt.legend()
plt.show()

index = np.arange(inliers.size)
cond_pred = inliers==1
actual = np.array([1 if i == 1 else -1 for i in df['LABEL']])


plt.figure()
plt.title('Predicted vs actual inliers exo planet dataset')
plt.scatter(index[cond_pred], inliers[cond_pred],color='g', marker='o',label='predicted inlier',alpha=0.2)
plt.scatter(index[~cond_pred], inliers[~cond_pred],color='r', marker='o',label='predicted outlier',alpha=0.2)
plt.plot(index, actual, color='b', label='actual')
plt.legend()
plt.ylim([-2, 2])
plt.show()