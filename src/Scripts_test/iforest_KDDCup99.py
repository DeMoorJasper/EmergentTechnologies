import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.decomposition import PCA

df = pd.read_csv('../data/kddcup.data.corrected.txt',header=None)

df.columns = ['duration','protocol_type','service','flag',
              'src_bytes','dst_bytes','land','wrong_fragment',
              'urgent','hot','num_failed_logins','logged_in',
              'num_compromised','root_shell','su_attempted','num_root',
              'num_file_creations','num_shells','num_access_files',
              'num_outbound_cmds','is_host_login','is_guest_login',
              'count','srv_count','serror_rate','srv_serror_rate',
              'rerror_rate','srv_rerror_rate','same_srv_rate',
              'diff_srv_rate','srv_diff_host_rate','dst_host_count',
              'dst_host_srv_count','dst_host_same_srv_rate','dst_host_diff_srv_rate',
              'dst_host_same_src_port_rate','dst_host_srv_diff_host_rate',
              'dst_host_serror_rate','dst_host_srv_serror_rate','dst_host_rerror_rate',
              'dst_host_srv_rerror_rate','label']

df = df[df.service == 'http']
df['label'] = pd.Series([0 if label=='normal.' else 1 for label in df['label']])
df = df.reset_index()
df = df.dropna()
contamination = min(df['label'].value_counts())/df['label'].size

# check which columns should be one-hot encoded and which numeric.
one_hots = ['flag','hot','logged_in']
numeric = ['duration','src_bytes','dst_bytes','num_compromised','root_shell','num_root','num_shells','num_access_files',
           'count','srv_count','serror_rate','srv_serror_rate','same_srv_rate','diff_srv_rate','srv_diff_host_rate','dst_host_count',
           'dst_host_srv_count','dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate','dst_host_srv_diff_host_rate',
           'dst_host_serror_rate','dst_host_srv_serror_rate','dst_host_rerror_rate','dst_host_srv_rerror_rate']

label = ['label']
#Do preprocessing.

#keep only the necessary collumns
df = df[one_hots + numeric + label]


# one hot encoding
df = pd.get_dummies(df, columns=one_hots)

# numeric normalization
scaler = MinMaxScaler()
df[numeric] = scaler.fit_transform(df[numeric])


#split df up in clean and malware subsets
df_clean = df[df['label']==0.]
df_malw = df[df['label']==1.]

df_complete = df.drop('label',1)
df_clean = df_clean.drop('label',1)
df_malw = df_malw.drop('label', 1)


#Perform PCA
pca = PCA(n_components=7)
df_complete_pca = pd.DataFrame(pca.fit_transform(df_complete))

#Get PCA versions clean and malware
df_complete_pca['label'] = df['label']
df_clean_pca = df_complete_pca[df_complete_pca['label']==0.]
df_malw_pca = df_complete_pca[df_complete_pca['label']==1.]

#Remove label collumns from PCA versions dataframes.
df_complete_pca = df_complete_pca.drop('label',1)
df_clean_pca = df_clean_pca.drop('label',1)
df_malw_pca = df_malw_pca.drop('label', 1)

#make Isolation forest with contamination rate equal to real contamination
Iforest = IsolationForest(n_estimators=300, max_samples=200, contamination=0.005)

#Do a fit on the entire dataset
Iforest.fit(df_complete_pca)

#Calculate scores for the different labels.
Scores_clean = Iforest.decision_function(df_clean_pca)
Scores_malw = Iforest.decision_function(df_malw_pca)

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


df.info()


