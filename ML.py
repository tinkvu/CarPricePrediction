import pickle
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

#import warnings
#warnings.filterwarnings("ignore")

df=pd.read_csv('modified_data.csv')

# Assuming 'X' contains your features and 'y' contains your target variable
X = df.drop('Selling Price', axis=1)
y = df['Selling Price']


df = np.array(df)

y = y.astype('int')
X = X.astype('int')
# print(X,y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = DecisionTreeRegressor(random_state=42)

# Training the model
model.fit(X_train, y_train)

pickle.dump(model,open('model.pkl','wb'))
