import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle
df = pd.read_excel('iris .xls')
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
df['class'] = label_encoder.fit_transform(df['Classification'])
x=df[['SL','SW','PL','PW']]
y=df['class']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,random_state=42)

lr = LinearRegression()
lr.fit(x_train, y_train)

pickle.dump(lr, open('model.pkl', 'wb'))
