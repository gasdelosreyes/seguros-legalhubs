import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder,LabelEncoder


'''
Import Dataset
'''

dataset = pd.read_csv('data.csv')
dataset = dataset.dropna()

X = dataset.iloc[:,3:-1].values
Y = dataset.iloc[:,-1].values

labelEncoder = LabelEncoder()
X[:,0] = labelEncoder.fit_transform(X[:,0])
X[:,1] = labelEncoder.fit_transform(X[:,1])
X[:,2] = labelEncoder.fit_transform(X[:,2])

# ct = ColumnTransformer([('encoder', OneHotEncoder())], remainder='passthrough') # The last arg ([0]) is the list of columns you want to transform in this step
# Y = ct.fit_transform(Y)
Y = pd.get_dummies(Y)

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=1)

standarizer = StandardScaler()
X_train = standarizer.fit_transform(X_train)
X_test = standarizer.fit_transform(X_test)

#Initialize model
ann = tf.keras.models.Sequential()

#Add an input layer
'''
Dense:  units = numbers of input neurons
        activation = activation function of the neurons
'''
ann.add(tf.keras.layers.Dense(units=3,activation='relu'))
ann.add(tf.keras.layers.Dense(units=3,activation='relu'))
ann.add(tf.keras.layers.Dense(units=4,activation='softmax'))
ann.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

ann.fit(X_train, Y_train, batch_size=32, epochs=150)

Y_pred = ann.predict(X_test)
print(Y_pred)