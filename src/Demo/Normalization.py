from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np


large_spread = np.random.normal(loc=1.0, scale=50, size=1000)
small_spread = np.random.normal(loc=1.5, scale=3, size=1000)



plt.figure()
plt.hist(large_spread,color='g', bins= 300)
plt.hist(small_spread, color='b', bins= 300)
plt.show()

scaler = MinMaxScaler()
large_spread = scaler.fit_transform(large_spread.reshape((-1, 1)))
small_spread = scaler.fit_transform(small_spread.reshape((-1, 1)))


plt.figure()
plt.hist(large_spread,color='g', bins= 300)
plt.hist(small_spread, color='b', bins= 300)
plt.show()

