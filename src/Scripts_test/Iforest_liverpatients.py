# This is an example of IsolationForest on the LiverPatient data set
# Anomalies are not few and different. In fact they are frequent!
# This doesn't produce satisfactory results.

from sklearn.ensemble.iforest import IsolationForest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df_patient = pd.read_csv('../data/indian_liver_patient.csv')

df_patient = df_patient.dropna()
df_patient =  df_patient.reset_index()

temp = df_patient.drop(axis=1,columns=['Dataset'])
temp = temp.dropna()
temp = temp.reset_index()

print(df_patient.info())
print(temp.info())


# Note that the internal treshold is set to
# expected contamination percentile of the scores.
# if contamination=0.1. Than lowest 10% of scores are considered outliers.
model = IsolationForest(max_samples=temp.shape[0]//5, n_estimators=temp.shape[0]//3,
                        random_state=2018, contamination=min(df_patient['Dataset'].value_counts())/temp.shape[0])
model.fit(temp)
scores = model.decision_function(temp)

inliers = model.predict(temp)
Label_count = df_patient['Dataset'].value_counts()
plt.figure()
plt.title('Scores liver patient dataset')
plt.hist(scores[inliers==1], bins=Label_count[1], label='healthy patient')
plt.hist(scores[inliers==-1], bins=Label_count[2], label='sick patient')
plt.legend()
plt.show()

index = np.arange(inliers.size)
cond_pred = inliers==1
cond_actual = np.array(df_patient['Dataset']==1)
ones = np.ones(inliers.size)
ones_neg = -1*ones

plt.figure()
plt.title('Predicted vs actual inliers liver patient dataset')
plt.scatter(index[cond_pred], inliers[cond_pred],color='g', marker='o',label='predicted inlier',alpha=0.2)
plt.scatter(index[~cond_pred], inliers[~cond_pred],color='r', marker='o',label='predicted outlier',alpha=0.2)
plt.plot(index[cond_actual], ones[cond_actual], color='b', label='actual inlier' )
plt.plot(index[~cond_actual], ones_neg[~cond_actual], color='b', label='actual outlier')
plt.legend()
plt.ylim([-2, 2])
plt.show()