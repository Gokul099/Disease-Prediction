import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)

df = pd.read_csv('Train_data.csv')
df.head()

"""# Basic checks"""

df.shape

df.info()

df.describe()

df.describe(include=['O'])

df.isnull().sum()

"""# There are no null values."""

df.duplicated().sum()

duplicated_rows = df[df.duplicated()]
print(duplicated_rows)

df.shape

"""# EDA"""

plt.figure(figsize = (15,10))
sns.countplot(data = df, x = 'Disease', palette = sns.color_palette('husl'))
plt.show()

"""## Anemia is more.
## Thromboc is less.
"""

plt.figure(figsize = (25,20))
sns.histplot(data = df,x = 'Glucose',hue = 'Disease')
plt.xlabel('Glucose',fontsize = 20)
plt.show()

df.columns

col = ['Green','Red','yellow','blue','black']

plt.figure(figsize = (8,6))
sns.scatterplot(data = df, x = 'Glucose', y = 'Cholesterol',hue = 'Disease', palette = col)
plt.show()

"""## If increasing in Glucose 0.1 - 0.4 and In Cholesterol 0.1 - 0.5 is level is healthy.
## if low level of Glucose and Cholesterol and high level of Glucose and Cholesterol is confirmally affected by Diabetes.
## And High Glucose and Low Cholestrol is affected Thalasse.
## Low Glucose and Low Cholesterol is affected Anemia.
"""

plt.figure(figsize = (8,6))
sns.scatterplot(data = df, x = 'Glucose', y = 'Hemoglobin',hue = 'Disease', palette = col)
plt.show()

"""## Glucose is high and hemoglobin is high is healthy.
## Glucose is low and hemoglobin is high is more affected Diabetes.
## Glucose is low and high hemoglobin are affected Thalasse.
## loe Glucose and low hemoglobin is affected Anemia.
## high Hemoglobin is affected Thromboc.
"""

plt.figure(figsize = (8,6))
sns.scatterplot(data = df, x = 'Insulin', y = 'Hemoglobin',hue = 'Disease', palette = col)
plt.show()

"""## high hemoglobin and normal and high insulin is chance to affected diabetes.
## high hemoglobin and low insulin is heathly.
## low and high insulin and high and low  hemoglobin is affected Anemia.
"""

plt.figure(figsize = (8,6))
sns.scatterplot(data = df, x = 'White Blood Cells', y = 'Red Blood Cells',hue = 'Disease', palette = col)
plt.show()

"""## normal Red blood cells and high white blood cell is makes you Healthy.
## Low red blood cells and high and normal white blood cells is affected Diabetes.
## high white blood cells and normal red blood cells is affeced Thalasse.
"""

plt.figure(figsize = (8,6))
sns.scatterplot(data = df, x = 'Insulin', y = 'BMI',hue = 'Disease', palette = col)
plt.show()

"""## normal and normal BMI and insulin is makes you healthy.
## low and high insulin and BMI is affected Anemia.
## high BMI is affected Thalasse.
"""

plt.figure(figsize = (8,6))
sns.scatterplot(data = df, x = 'Cholesterol', y = 'BMI',hue = 'Disease', palette = col)
plt.show()

"""## high cholesrerol and normal BMI is makes you healthy.
## high BMI and medium cholesterol is affected Thalasse.
## low and medium of cholesterol and BMI is affected Animea.

# Encoding
"""

df.Disease.value_counts()

df['Disease'].replace({'Healthy':1,'Diabetes':2,'Anemia':3,'Thalasse':4,'Thromboc':5}, inplace = True)

df.Disease.value_counts()

"""# Split"""

s = df.corr()
print(s)

x = df.drop('Disease',axis = 1)
y = df['Disease']

x.head()

y

"""# spliting"""

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 42)

x_train.shape

y_test.shape

"""# model creation"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, make_scorer
rc = RandomForestClassifier()
rc.fit(x_train,y_train)

x1_pred = rc.predict(x_train)
x1_pred

y1_pred = rc.predict(x_test)
y1_pred

# Accuracy
print(f'TR:{classification_report(x1_pred,y_train)}')
print(f'TS:{classification_report(y1_pred,y_test)}')

from sklearn.model_selection import GridSearchCV

para = {'n_estimators' : [100,200,300,600],
        'min_samples_split':  [0.1,0.2,0.3],
        'max_depth' :   [None,1,2,3],
        'min_samples_leaf' : [None,0.1,1,2],
        'max_features'  :   ['sqrt','log2']
       }

model1 = RandomForestClassifier()
Gs = GridSearchCV(model1,para,cv = 2,n_jobs = -1)
Gs.fit(x_train,y_train)

best = Gs.best_params_
print(best)

m1 = RandomForestClassifier(max_depth = None, max_features = 'sqrt', min_samples_leaf = 1, min_samples_split = 0.1,
                            n_estimators = 100, random_state = 42)
m1.fit(x_train,y_train)

x2 = m1.predict(x_train)
x2

y2 = m1.predict(x_test)
y2

# Accuracy
print(f'TR:{classification_report(y_train,x2)}')
print(f'TS:{classification_report(y_test,y2)}')

import joblib

file = 'Disease prediction'
joblib.dump(m1,'Disease prediction')
app = joblib.load('Disease prediction')
arr = [[0.739596713,0.650198388,0.713630986,0.868491241,0.687433028,0.529895399,0.290005909,
        0.631045018,0.001327858,0.79582887,0.034129122,0.071774199,0.185595597,0.07145461,0.653472376,
        0.502664779,0.215560238,0.512940563,0.064187347,0.61082651,0.939484854,0.095511528,0.465956967,0.769230075
]]
d = app.predict(arr)
print(d)

import pickle

Pkl_Filename = "Pickle_RL_Model.pkl"  


with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(m1, file)


with open(Pkl_Filename, 'rb') as file:  
    Pickled_Model = pickle.load(file)

