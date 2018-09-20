import pandas as pd
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

for i in df.columns:
    print('column.value_counts()', i , 'has shape:', df[i].value_counts().shape)

# following collumnns need to be processed rest are garbage
# one_hots = ['flag','hot','logged_in']
# numerical = ['duration','src_bytes','dst_bytes','num_compromised','root_shell','num_root','num_shells','num_access_files',
#              'count','srv_count','serror_rate','srv_error_rate','same_srv_rate','diff_srv_rate','srv_diff_host_rate','dst_host_count'
#              'dst_host_srv_count','dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate','dst_host_srv_diff_host_rate',
#              'dst_host_serror_rate','dst_host_srv_serror_rate','dst_host_rerror_rate','dst_host_srv_rerror_rate']

#make a PCA object that retains 95% of the variance of the dataset
pca = PCA(0.95)

#apply PCA to df
pca.fit(df)
df = pca.transform(df)
df.info()

