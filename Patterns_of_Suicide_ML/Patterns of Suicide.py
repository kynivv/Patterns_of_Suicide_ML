import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

import warnings
warnings.filterwarnings('ignore')


# Data Import
df = pd.read_csv('Suicide.csv')
df = df.drop(['method'], axis=1)


# EDA
#print(df.describe)
#print(df.dtypes)
#print(df.isnull().sum())

features = ['sex', 'Freq']
sns.displot(x=df[features[1]], y=df[features[0]],color='pink')
#plt.show()


# Data Transformation
for col in df.columns :
    if df[col].dtype == 'object':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])


# Features Correlation
sns.heatmap(df.corr(), cbar= False, annot= True)
#plt.show()


# Data Splitting
from sklearn.model_selection import train_test_split

features = df.drop('Freq', axis=1)
target = df['Freq']

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.23, random_state=22)
#print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# Model Training
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import explained_variance_score as evs

models = [LinearRegression(), RandomForestRegressor(), XGBRegressor()]

for m in models:
    m.fit(X_train, y_train)

    pred_train = m.predict(X_train)
    print(f'Train Accuracy of {m} is : {(evs(y_train, pred_train))}')

    pred_test = m.predict(X_test)
    print(f'Test Accuracy of {m} is : {(evs(y_test, pred_test))}')

### XGBRegressor shows the best results ;) 





