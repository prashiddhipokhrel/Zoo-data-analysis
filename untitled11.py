import os 
import pandas as pd
import numpy as np 
import matplotlib.pyplot as pyplot
import seaborn as sb 
from sklearn.preprocessing import StandardScaler as SS 
import plotly.graph_objects as go
import plotly.offline as py
import csv
import plotly.express as px

os.getcwd()
path = ('C:/Users/Swift/OneDrive/Desktop/Data Science/Data3/Zoo')
os.chdir(path)

data = pd.read_csv("zoo.csv")
data.head()
data.columns 
len(data.columns) ## 18 

## cleaning the data ### 
data.isnull().sum().sum()
data.isna().sum().sum()
## both is equal to 0 ##

## check if there is any duplicated columns ## 
data.columns.duplicated().sum()

## cheking the first column ## 
data['animal_name'].value_counts().sum()
## it looks like all of the names are different ## 

## Too many different names in animal names, so I think it is appropriate to take
## class type as the response variable and others as the predictor variable ## 

## checking the last column for the class types ### 
data['class_type'].value_counts()
# Out[77]: 
# 1    41
# 2    20
# 4    20
# 6    20
# 3    17
# 7    16
# 5    10
# Name: class_type, dtype: int64

### seperating x and y for the data set ###
y = data.iloc[:,17:18]
x = data.iloc[:, 1:17]

## scalling the variables to make them workable ## 
scaler = SS()
x = scaler.fit_transform(x)
x = scaler.transform(x)

### generating a histogram for the class types ### 
axx = data['class_type'] 
sb.set_style('darkgrid')
ax = sb.countplot(axx , palette = 'magma')
pyplot.xlabel('Classes')
pyplot.title('Frequency of class types')
pyplot.show()

## summarized table ## 
summary = data.describe().T 
summary = summary.round(4)
summary 
# Out[109]: 
#             count    mean     std  min   25%  50%   75%  max
# hair        144.0  0.3056  0.4623  0.0  0.00  0.0  1.00  1.0
# feathers    144.0  0.1389  0.3470  0.0  0.00  0.0  0.00  1.0
# eggs        144.0  0.7083  0.4561  0.0  0.00  1.0  1.00  1.0
# milk        144.0  0.2847  0.4529  0.0  0.00  0.0  1.00  1.0
# airborne    144.0  0.2153  0.4124  0.0  0.00  0.0  0.00  1.0
# aquatic     144.0  0.3889  0.4892  0.0  0.00  0.0  1.00  1.0
# predator    144.0  0.4792  0.5013  0.0  0.00  0.0  1.00  1.0
# toothed     144.0  0.5556  0.4986  0.0  0.00  1.0  1.00  1.0
# backbone    144.0  0.7500  0.4345  0.0  0.75  1.0  1.00  1.0
# breathes    144.0  0.7847  0.4124  0.0  1.00  1.0  1.00  1.0
# venomous    144.0  0.0903  0.2876  0.0  0.00  0.0  0.00  1.0
# fins        144.0  0.1667  0.3740  0.0  0.00  0.0  0.00  1.0
# legs        144.0  2.9514  2.2104  0.0  0.00  4.0  4.00  8.0
# tail        144.0  0.6667  0.4730  0.0  0.00  1.0  1.00  1.0
# domestic    144.0  0.1250  0.3319  0.0  0.00  0.0  0.00  1.0
# catsize     144.0  0.4167  0.4947  0.0  0.00  0.0  1.00  1.0
# class_type  144.0  3.4306  2.1374  1.0  1.00  3.0  5.25  7.0

## correlation matrix ### 
data = data.drop(['animal_name'], axis = 1)

### correlation matrix ### 
fig, ax = pyplot.subplots(figsize = (20,10))
sb.heatmap(data.corr(), annot=True, fmt='.1g', cmap="viridis",
           cbar=False, linewidths=0.7, linecolor='black');

### looking at the correlation matrix, we can see strong positive correlation between
### milk and hair,

## fair strong positive between  airborne and feathers, 


### strong negative correlation between milk and eggs , hair and eggs

## we can also see that the class_type has failry strong positve relation 
##with eggs and failry strong neagative relation with milk, backbone, tail ### 
## kind of see that these are might be the stronger predecitors of the class types ##

#### lets use linear regression ### 
## seperate the x and y to test and train sets ### 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

#### LINEAR REGRESSION #### 
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)

LinearRegressionScore = lr.score(X_test, y_test)
print("Accuracy obtained by Linear Regression model:",LinearRegressionScore*100) 
### Accuracy score 96.99% ###


#### LOGISTIC REGRESSION ### 
from sklearn.linear_model import LogisticRegression 
model = LogisticRegression()
model.fit(X_train, y_train)

score = model.score(X_test, y_test)
print("Accuracy obtained by Logistic Regression model:",score*100) 
### Accuracy score is 97.22% ### 

### CROSS VALIDATION ## 
X_train, X_test, y_train, y_test = train_test_split(x,y, test_size = 0.50,
                                                    random_state = 1)
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score
kfold = KFold(n_splits = 50, random_state = 7)
model = LogisticRegression()
results = cross_val_score(model,x,y,cv = kfold)
print(results) 
print("Accuracy obtained by Logistic Regression model:",results.mean*100)
## Accuracy score is : ####

### Non-Linear Methods ### 
from sklearn import svm 
import sklearn.model_selection as model_selection
from sklearn.metrics import accuracy_score
X_train, X_test, y_train, y_test = model_selection.train_test_split(x,
                    y, train_size=0.80, test_size=0.20, random_state=101)
rbf = svm.SVC(kernel='rbf', gamma=0.5, C=0.1).fit(X_train, y_train)
rbf_pred = rbf.predict(X_test)
rbf_accuracy = accuracy_score(y_test, rbf_pred)
print("Accuracy obtained multiclass SVM :",rbf_accuracy *100)
### 41.4 % accuracy ### 

## KNN ### 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x,
                  y, test_size=0.2, random_state=12345)
from sklearn.neighbors import KNeighborsRegressor
knn_model = KNeighborsRegressor(n_neighbors=3)
knn_model.fit(X_train, y_train)
knn_pred = knn_model.predict(X_test)
knn_accuracy = accuracy_score(y_test, knn_pred)

### MLP ### 
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, stratify=y,
                                                    random_state=1)
clf = MLPClassifier(random_state=1, max_iter=300).fit(X_train, y_train)
clf.predict(X_test)
score = clf.score(X_test, y_test)
print("Accuracy obtained MLP :",score*100)
### accuracy score is 91.67 ### 

## Hypothesis testing ### 
## null hypothesis mean is 0 #### 
from scipy import stats 
from statsmodels.stats import weightstats as stests

ztest, pval = stests.ztest(data['breathes'], x2 = None, value = 0, 
                           alternative = 'two-sided')
print(float(pval))

if pval<0.05:
    print("reject null hypothesis")
else:
    print("accept null hypothesis")
    
## we reject the null hypothesis that the mean is zero,two sided test ### 


