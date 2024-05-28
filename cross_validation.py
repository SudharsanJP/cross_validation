#)importing necessary modules
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier,GradientBoostingClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

st.title(':orange[cross validation ML project]')
st.write(":green[**GUVI**]")
#)reading the dataset
original_df = pd.read_csv(r"D:\Sudharsan\Guvi_Data science\DS101_Sudharsan\Mainboot camp\data\loan_approval_dataset.csv")

#)dummies
cols = [' education',' self_employed',' loan_status']
df = pd.get_dummies(original_df,columns=cols,drop_first='True',dtype='int')

st.title(':violet[1.cross validations]')
selectBox=st.selectbox("iteration: ", ['dataset',
                                       'kfold',
                                       'skfold',
                                       'shufflesplit'
                                       ])
if selectBox == 'dataset':
    st.markdown("\n#### :blue[1.1 dataset:]")
    st.dataframe(original_df.head(6))

elif selectBox == 'kfold':
   st.markdown("\n#### :blue[1.2 kfold:]")
   from sklearn.model_selection import KFold, cross_val_score
   #) X and y
   X =df.drop([' loan_status_ Rejected'],axis=1)
   y = df[' loan_status_ Rejected']
   
   #)splitting training and testing data
   x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
   
   #) Ml models
   models = [LogisticRegression(),
             KNeighborsClassifier(),
             SVC(),
             DecisionTreeClassifier(),
             RandomForestClassifier(),
             AdaBoostClassifier(),
            GradientBoostingClassifier()
            ]
   
   k_folds = KFold(n_splits = 5)
   
   for model in models:
    st.success(model)
    scores = cross_val_score(model, X, y, cv = k_folds)
    st.write(f":red[Cross Validation Scores:  {scores}]")
    st.write(f":green[Average CV Score: {scores.mean()}]")
    st.write(f":orange[Number of CV Scores used in Average: {len(scores)}]")

#) skfold
elif selectBox == 'skfold':
   st.markdown("\n#### :blue[1.3 skfold:]")
   from sklearn.model_selection import StratifiedKFold, cross_val_score
   
   #) X and y
   X =df.drop([' loan_status_ Rejected'],axis=1)
   y = df[' loan_status_ Rejected']
   
   #)splitting training and testing data
   x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
   
   #) Ml models
   models = [LogisticRegression(random_state=42),
             KNeighborsClassifier(),
             SVC(random_state=42),
             DecisionTreeClassifier(random_state=42),
             RandomForestClassifier(random_state=42),
             AdaBoostClassifier(random_state=42),
             GradientBoostingClassifier(random_state=42)
             ]
   
   sk_folds = StratifiedKFold(n_splits = 5)

   for model in models:
    st.success(model)
    scores = cross_val_score(model, X, y, cv = sk_folds)
    st.write(f":red[Cross Validation Scores:  {scores}]")
    st.write(f":green[Average CV Score: {scores.mean()}]")
    st.write(f":orange[Number of CV Scores used in Average: {len(scores)}]")

#)shuffle split
elif selectBox == 'shufflesplit':
   st.markdown("\n#### :blue[1.4 shuffle split:]")
   from sklearn.model_selection import ShuffleSplit, cross_val_score
   
   #) X and y
   X =df.drop([' loan_status_ Rejected'],axis=1)
   y = df[' loan_status_ Rejected']
   
   #)splitting training and testing data
   x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
   
   #) Ml models
   models = [LogisticRegression(random_state=42),
             KNeighborsClassifier(),
             SVC(random_state=42),
             DecisionTreeClassifier(random_state=42),
             RandomForestClassifier(random_state=42),
             AdaBoostClassifier(random_state=42),
             GradientBoostingClassifier(random_state=42)
             ]
   
   ss = ShuffleSplit(train_size=0.6, test_size=0.3, n_splits = 5)
   
   for model in models:
    st.success(model)
    scores = cross_val_score(model, X, y, cv = ss)
    st.write(f":red[Cross Validation Scores:  {scores}]")
    st.write(f":green[Average CV Score: {scores.mean()}]")
    st.write(f":orange[Number of CV Scores used in Average: {len(scores)}]")