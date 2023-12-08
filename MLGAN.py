


import pandas as pd
df=pd.read_csv('flags.csv')
features=19
X = df.iloc[:, :features]
y = df.iloc[:, features:]
discrete_columns=df.columns[features:]


import sklearn.metrics

from IPython.display import display, HTML, Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy
import random
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import make_scorer
from sklearn import metrics
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import KFold
kf = KFold(n_splits=5, random_state=None, shuffle=True) # Define the split - into 2 folds 
kf.get_n_splits(X) # returns the number of splitting iterations in the cross-validator
print(kf) 
t=1
for train_index, test_index in kf.split(X):
    print('TRAIN:', train_index, 'TEST:', test_index)
    
 
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    df_train=pd.concat([X_train,y_train],axis=1)
    # Check Imbalance
    ones=[]
    for i in list(y_train.columns):
        L=df_train[i].value_counts()
        ones.append(L[1])
        # ones.append(L[1]) for flags
    maxx=max(ones)

    irlbl=[]
    for i in ones:
        div=maxx/i
        irlbl.append(div)


    import numpy as np
    mean=np.mean(irlbl)

    imb=[]
    for i in irlbl:
        if(i>mean):
            imb.append(i)
        
    d=maxx/mean

    items=[]
    for i in ones:
        l = d-i
        items.append(l)
        
    items = list(map(int, items))
    items=[max(num, 0) for num in items]  

    syn=[]
    k=0
    from ctgan import CTGANSynthesizer
    ctgan = CTGANSynthesizer(epochs=80)
    for i in list(y.columns):
        df_train1=df_train.loc[df_train[i] == 1]
        # df_train1=df_train.loc[df_train[i] == 1] for flags
        # df_train1=df_train1.sample(frac = 0.4)
        ctgan.fit(df_train1, discrete_columns)
        for j in items:
            samples = ctgan.sample(items[k])
            break
        k=k+1
        syn.append(samples)
        print("loop")
    
    sampless=pd.DataFrame()
    for i in range(len(discrete_columns)):
        p=syn[i]
        sampless=pd.concat([sampless,p],axis=0)
        
    
    
    
    
    train=pd.concat([df_train,sampless],axis=0)
    X_train = train.iloc[:, :features]
    y_train = train.iloc[:, features:]
    
    # using Label Powerset
    from skmultilearn.problem_transform import LabelPowerset
    # initialize label powerset multi-label classifier
    classifier = LabelPowerset(SVC())
    # train
    classifier.fit(X_train, y_train)
    # predict
    y_pred = classifier.predict(X_test)
    y_pred=pd.DataFrame.sparse.from_spmatrix(y_pred)
    y_pred=y_pred.set_axis([i for i in list(discrete_columns)], axis=1, inplace=False)




    print("Hamming Loss", sklearn.metrics.hamming_loss(y_test, y_pred))
        
    from sklearn.metrics import accuracy_score
    #MR = np.all(y_pred == y_test, axis=0).mean()
    print("Exact Match Ratio:", accuracy_score(y_test, y_pred))
        
    # zero_one_Loss = np.any(y_test != y_pred, axis=0).mean()
    #print("0/1 Loss:", 1-accuracy_score(y_test, y_pred))
        
    #print("Accuracy Score:", 1-sklearn.metrics.hamming_loss(y_test, y_pred))
        
    #print('Precision:', sklearn.metrics.recall_score(y_true=y_test, y_pred=y_pred, average='samples'))

   # print("Recall:", sklearn.metrics.precision_score(y_true=y_test, y_pred=y_pred, average='samples'))
        
    print("F1 Measure:", sklearn.metrics.f1_score(y_true=y_test, y_pred=y_pred, average='samples'))
        
    from sklearn.metrics import label_ranking_loss
    print("Ranking Loss:", label_ranking_loss(y_test, y_pred.to_numpy()))
        
    from sklearn.metrics import roc_auc_score
    try:
        print("ROC-AUC:", roc_auc_score(y_test,y_pred.to_numpy(),multi_class='ovr'))
    except ValueError:
        pass
    
    #test=pd.concat([X_test,y_test],axis=1)
    #df_new=pd.concat([df_train,sampless,test],axis=0)
    #df_new.drop(df_new.columns[1], axis=1)
    #df_new.to_csv('flags_LabelPowerSet_SVC.csv', index=False)
     