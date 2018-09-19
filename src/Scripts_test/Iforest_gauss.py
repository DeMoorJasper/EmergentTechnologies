from sklearn.ensemble.iforest import IsolationForest
import numpy as np
import matplotlib.pyplot as plt


# Create 500 points for normal distributions with average at -2 and 2
# and deviance 0.5 .
X = np.concatenate((np.random.normal(loc=-2, scale=.5, size=500),
                    np.random.normal(loc=2, scale=.5,  size=500)))

# make histogram plots of the random generated values.
plt.hist(X, normed=True)
plt.xlim([-5, 5])
plt.show()

#make Iforest model and construct Itrees on generated data
isolation_forest = IsolationForest(n_estimators=100, max_samples=300)
isolation_forest.fit(X.reshape(-1, 1))
xx = np.arange(-6, 6, 12/100).reshape(-1,1)

#Calculate anomaly scores and predict outliers for range -6 to 6
anomaly_score = isolation_forest.decision_function(xx)
outlier = isolation_forest.predict(xx)

#plot anomaly scores
plt.figure()
plt.plot(xx, anomaly_score, label='anomaly score')

#color region where anomalies are predicted red.
y = np.zeros(xx.shape[0])
y.fill(max(anomaly_score))
mask = outlier == -1
plt.scatter(xx[mask], y[mask], color='r', label='outliers')
plt.show()
plt.fill_between(xx.T[0], np.min(anomaly_score), np.max(anomaly_score),
                 where=outlier == -1, color='r',
                 alpha=.4, label='outlier region')
plt.legend()
plt.ylabel('anomaly score')
plt.xlabel('X')
plt.xlim([-5, 5])
plt.show()

