import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

rng = np.random.RandomState(42)

# Generate train data
X = 0.3 * rng.randn(1000, 2)
X_train = np.r_[X + 2, X - 2]
# Generate some regular novel observations
X = 0.3 * rng.randn(200, 2)
X_test = np.r_[X + 2, X - 2]
# Generate some abnormal novel observations
X_outliers = rng.uniform(low=-4, high=4, size=(200, 2))

# fit the model
clf = IsolationForest(n_estimators=300,max_samples=300, random_state=rng)
clf.fit(X_train)
y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)
y_pred_outliers = clf.predict(X_outliers)


plt.title("IsolationForest")

b1 = plt.scatter(X_train[:, 0], X_train[:, 1], c='white',
                 s=20, edgecolor='k')
b2 = plt.scatter(X_test[:, 0], X_test[:, 1], c='green',
                 s=20, edgecolor='k')
c = plt.scatter(X_outliers[:, 0], X_outliers[:, 1], c='red',
                s=20, edgecolor='k')
bool_train = y_pred_train == -1
bool_test = y_pred_test == -1
bool_outlier = y_pred_outliers == -1
plt.scatter(X_train[bool_train, 0], X_train[bool_train, 1], c='black',alpha=0.5,
            s=20,marker='x')
plt.scatter(X_test[bool_test, 0], X_test[bool_test, 1],c='black',alpha=0.5,
            s=20, marker='x')
plt.scatter(X_outliers[bool_outlier, 0], X_outliers[bool_outlier, 1], c='black',alpha=0.5,
            s=20, marker='x')

plt.axis('tight')
plt.xlim((-5, 5))
plt.ylim((-5, 5))
plt.legend([b1, b2, c],
           ["training observations",
            "new regular observations", "new abnormal observations"],
           loc="upper left")
plt.show()