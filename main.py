import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
import time
plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns

sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

data = pd.read_csv('//Users/shravankaul/Desktop/UWM/Stat 601/Project 2/Project2Data/dat.csv', header=0,encoding= 'unicode_escape')
data = data.dropna()
print(data.shape)
print(list(data.columns))
print(data['MajLang'].unique())
data['MajLang'] = np.where(data['MajLang'] == 'R', 1, data['MajLang'])
data['MajLang'] = np.where(data['MajLang'] != 1, 0, data['MajLang'])
print(data['MajLang'].unique())
sns.countplot(x='MajLang', data=data, palette='hls')
#plt.show()
count_not_R = len(data[data['MajLang'] == 0])
count_R = len(data[data['MajLang'] == 1])
pct_of_not_R = count_not_R / (count_not_R + count_R)
print("percentage of not R", pct_of_not_R * 100)
pct_of_R = count_R / (count_not_R + count_R)
print("percentage of R", pct_of_R * 100)

tags_c=data['tags']
data['tags'] = data['tags'].str.replace(',','')
data['tags'] = data['tags'].str.replace('[','')
data['tags'] = data['tags'].str.replace(']','')
data['tags'] = data['tags'].str.replace("'","")
data['tags']=data['tags'].str.split()
data['Descr']=data['Descr'].str.encode('ascii', 'ignore').str.decode('ascii')

from sklearn.preprocessing import MultiLabelBinarizer
mlb = MultiLabelBinarizer()
df=pd.DataFrame(data=data)
mlb = MultiLabelBinarizer(sparse_output=True)

df = df.join(
            pd.DataFrame.sparse.from_spmatrix(
                mlb.fit_transform(df.pop('tags')),
               index=df.index,
                columns=mlb.classes_))


df['tags_c']=tags_c
df['Lang-Descr']= None
Lang_Descr=df['Lang-Descr']
df.drop(labels=['Lang-Descr'], axis=1,inplace = True)
df.insert(6, 'Lang-Descr', Lang_Descr)

df.drop(labels=['tags_c'], axis=1,inplace = True)
df.insert(4, 'tags_c', tags_c)



df=df.reset_index()
print(len(df['Descr']))
import pycld2 as cld2
for i in range(len(df['Descr'])) :
    _, _, _, detected_language = cld2.detect(df['Descr'][i],  returnVectors=True)
    if len(detected_language)==0:
        df['Lang-Descr'][i]=0
    else:
        df['Lang-Descr'][i]= detected_language[0][2]

df=df[df['Lang-Descr'] != 0]

print(df['Lang-Descr'].head())

df.to_csv('//Users/shravankaul/Desktop/test/sk3.csv')

