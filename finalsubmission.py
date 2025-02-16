import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_squared_error, mean_absolute_error
train_data = pd.read_csv('train_v9rqX0R.csv')
test_data = pd.read_csv('test_AbJTz2l.csv')
train_data['Item_Weight'] = train_data['Item_Weight'].fillna(train_data['Item_Weight'].median())
cols = ['Outlet_Identifier', 'Outlet_Establishment_Year', 'Outlet_Location_Type', 'Outlet_Type', 'Outlet_Size']
df_selected = train_data[cols]
label_encoders = {}
for col in ['Outlet_Identifier', 'Outlet_Location_Type', 'Outlet_Type']:
    le = LabelEncoder()
    df_selected[col] = le.fit_transform(df_selected[col])
    label_encoders[col] = le
df_missing = df_selected[df_selected['Outlet_Size'].isnull()].copy()
df_complete = df_selected.dropna().copy()
ohe = OneHotEncoder(sparse_output=False, drop='first')
outlet_size_encoded = ohe.fit_transform(df_complete[['Outlet_Size']])
clf = RandomForestClassifier()
X_train = df_complete.drop(columns=['Outlet_Size'])
y_train = outlet_size_encoded
clf.fit(X_train, y_train)
X_missing = df_missing.drop(columns=['Outlet_Size'])
predicted = clf.predict(X_missing)
predicted_labels = ohe.inverse_transform(predicted)
train_data.loc[train_data['Outlet_Size'].isnull(), 'Outlet_Size'] = predicted_labels
test_data['Item_Weight'] = test_data['Item_Weight'].fillna(test_data['Item_Weight'].median())
df_selected = test_data[cols]
label_encoders = {}
for col in ['Outlet_Identifier', 'Outlet_Location_Type', 'Outlet_Type']:
    le = LabelEncoder()
    df_selected[col] = le.fit_transform(df_selected[col])
    label_encoders[col] = le
df_missing = df_selected[df_selected['Outlet_Size'].isnull()].copy()
df_complete = df_selected.dropna().copy()
ohe = OneHotEncoder(sparse_output=False, drop='first')
outlet_size_encoded = ohe.fit_transform(df_complete[['Outlet_Size']])
clf = RandomForestClassifier()
X_train = df_complete.drop(columns=['Outlet_Size'])
y_train = outlet_size_encoded
clf.fit(X_train, y_train)
X_missing = df_missing.drop(columns=['Outlet_Size'])
predicted = clf.predict(X_missing)
predicted_labels = ohe.inverse_transform(predicted)
test_data.loc[test_data['Outlet_Size'].isnull(), 'Outlet_Size'] = predicted_labels
train_data['Item_Fat_Content'] = train_data['Item_Fat_Content'].replace({'low fat': 'Low Fat', 'LF':'Low Fat', 'reg':'Regular'})
test_data['Item_Fat_Content'] = test_data['Item_Fat_Content'].replace({'low fat': 'Low Fat', 'LF':'Low Fat', 'reg':'Regular'})
train_data['Item_Visibility'] = train_data.groupby('Item_Identifier')['Item_Visibility'].transform(lambda x : x.replace(0, x.mean()))
test_data['Item_Visibility'] = test_data.groupby('Item_Identifier')['Item_Visibility'].transform(lambda x : x.replace(0, x.mean()))
train_data['Item_Fat_Content'] = train_data['Item_Fat_Content'].map({'Low Fat' : 0, 'Regular': 1})
train_data['Outlet_Size'] = train_data['Outlet_Size'].map({'Small':0, 'Medium': 1, "High": 2})
train_data['Outlet_Location_Type'] = train_data['Outlet_Location_Type'].map({'Tier 3': 0 , 'Tier 2': 1, 'Tier 1': 2})
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
train_data['Outlet_Type'] = le.fit_transform(train_data['Outlet_Type'])
train_data['Age_of_store'] = 2013 - train_data['Outlet_Establishment_Year']
test_data['Item_Fat_Content'] = test_data['Item_Fat_Content'].map({'Low Fat' : 0, 'Regular': 1})
test_data['Outlet_Size'] = test_data['Outlet_Size'].map({'Small':0, 'Medium': 1, "High": 2})
test_data['Outlet_Location_Type'] = test_data['Outlet_Location_Type'].map({'Tier 3': 0 , 'Tier 2': 1, 'Tier 1': 2})
test_data['Outlet_Type'] = le.fit_transform(test_data['Outlet_Type'])
test_data['Age_of_store'] = 2013 - test_data['Outlet_Establishment_Year']
X  = train_data.drop(columns= ['Item_Outlet_Sales', 'Outlet_Establishment_Year', 'Outlet_Identifier', 'Item_Type', 'Item_Identifier'])
y  = train_data['Item_Outlet_Sales']
X_test = test_data.drop(columns=['Item_Identifier', 'Item_Type', 'Outlet_Identifier', 'Outlet_Establishment_Year'])
sc = StandardScaler()
X_train = sc.fit_transform(X)
X_test = sc.transform(X_test)
model = Sequential()
model.add(Dense(64, input_dim = X_train.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))  
model.add(Dense(1, activation = 'relu')) 

model.compile(optimizer='adam', loss='huber_loss', metrics=['mae'])

history = model.fit(X_train, y, epochs=100,batch_size=32)
predictions = model.predict(X_test).flatten()
predictions = predictions.flatten()
non_negative_predictions = [max(pred, 0) for pred in predictions]
submission = pd.DataFrame({
    'Item_Identifier': test_data['Item_Identifier'],
    'Outlet_Identifier': test_data['Outlet_Identifier'],
    'Item_Outlet_Sales': predictions
})
print(submission.shape)
submission.to_csv('output1.csv', index=False)