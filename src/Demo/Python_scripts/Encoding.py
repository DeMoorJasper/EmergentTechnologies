from sklearn.preprocessing import LabelEncoder
import pandas as pd

#LabelEncoding
Genders = ['male','female','X','male','female','female','male','female','Y']
df_gender = pd.DataFrame(Genders)
df_gender.columns = ['Genders']
print(df_gender.head())

encoder = LabelEncoder()
df = encoder.fit_transform(df_gender['Genders'])
print(df)

Temperatures = ['HyperThermia', 'HypoThermia','Body', 'Death high T', 'Death low T','Body',
                'HyperThermia','HypoThermia','Death high T']
df_temp = pd.DataFrame(Temperatures)
df_temp.columns = ['Temp']
df = encoder.fit_transform(df_temp['Temp'])
print(df)

df = pd.get_dummies(df_gender, columns=['Genders'])
print(df)
print(df_gender)

df = pd.get_dummies(df_temp, columns=['Temp'])
print(df)
print(df_gender)







