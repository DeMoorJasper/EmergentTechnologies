import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.decomposition import PCA


#Clean data
df = pd.read_csv('../data/creditcard.csv')
df = df.dropna()  # drop rows with Nan
df = df.reset_index()
df_dropped = df.drop(['Class','index','Time'], 1)


#Calculate contamination
contamination = min(df['Class'].value_counts())/df['Class'].size

# numeric normalization
scaler = MinMaxScaler()
df_dropped = scaler.fit_transform(df_dropped)

#get clean, dirty subsets
df_clean = df_dropped[df['Class'] == 0.]
df_malw = df_dropped[df['Class'] == 1.]

# boot the iforest object
Iforest = IsolationForest(n_estimators=300, max_samples=300, contamination=0.005)

#Do a fit on the entire dataset
Iforest.fit(df_dropped)


#Calculate scores for the different labels.
Scores_clean = Iforest.decision_function(df_clean)
Scores_malw = Iforest.decision_function(df_malw)

#plot the scores in a histogram
plt.figure()
plt.title("Scores Iforest Normal")
plt.hist(Scores_clean, bins=100, color='g', alpha=0.5, label='scores clean')
plt.axvline(x=Iforest.threshold_, color='orange', alpha=0.2)
plt.legend(loc='upper right')
plt.show()

plt.figure()
plt.title('Scores Iforest Malware')
plt.hist(Scores_malw, bins=100, color='r', alpha=0.5, label='scores dirty')
plt.legend(loc='upper right')
plt.show()

#Make prediction on inliers for the complete dataset
inliers = Iforest.predict(df_dropped)

#plot predicted inliers vs. actual malware points.
index = np.arange(df_dropped.shape[0])
y = np.ones(df_dropped.shape[0])
mask = (inliers == 1)
plt.figure()
plt.title('predicted Inliers vs. Actual Inliers')
plt.ylim(0.9, 1.6)
plt.plot(index[mask],y[mask],color='g',label='non_malware_pred' )
plt.scatter(index[~mask], y[~mask], color='r', label='malware_pred',
            marker='o', s=2)
plt.scatter(index[df['Class']==1.], 1.1*y[df['Class']==1.], color='y', label='malware', marker = 'o',
            s=2, alpha=0.5)
plt.legend(loc='upper right')
plt.show()
df.info()


