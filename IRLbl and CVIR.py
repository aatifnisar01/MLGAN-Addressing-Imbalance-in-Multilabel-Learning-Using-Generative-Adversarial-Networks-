import pandas as pd
df=pd.read_csv('mirflickr.csv')

features=150

X = df.iloc[:, :features]
y = df.iloc[:, features:]
discrete_columns=df.columns[features:]

#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#df=pd.concat([X_train,y_train],axis=1)

ones=[]
for i in list(y.columns):
    L=df[i].value_counts()
    ones.append(L[1])
    
maxx=max(ones)

irlbl=[]
for i in ones:
    div=maxx/i
    irlbl.append(div)


import numpy as np
mean=np.mean(irlbl)

x1=1/mean
x2=0
for i in irlbl:
    z=(i-mean)**2
    x2+=z
x3=len(discrete_columns)
x4=x2/(x3-1)
x5=x4**0.5
cvir=x1*x5
print(cvir)
