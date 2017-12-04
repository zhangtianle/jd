from sys import path
path.append('../../')
from keras.layers import Dense, Dropout
from keras.models import Sequential
import pandas as pd
from sklearn.model_selection import train_test_split

from tl.src.Main import data_scaler, delete

X = pd.DataFrame(pd.read_csv("../feature/train_x_offline_start_8_end_10.csv"))
Y = pd.DataFrame(pd.read_csv("../feature/train_y_11_offline.csv"))
Test = pd.DataFrame(pd.read_csv("../feature/train_x_offline_start_9_end_11.csv"))

X = X.fillna(0)
Test = Test.fillna(0)

_, uid = delete(X, Test, "uid")
# delete(X, Test, "average_discount")

Y.pop("uid")

train_X = X.as_matrix()
train_Y = Y.as_matrix()

train_X = data_scaler(train_X)

X_train, X_test, y_train, y_test = train_test_split(train_X, train_Y, test_size=0.2, random_state=1)

model = Sequential()
model.add(Dense(40, input_shape=(len(X_train[0]),), activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(8, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='relu'))
model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, batch_size=16, epochs=400, validation_data=(X_test, y_test), verbose=2)
y = model.predict(X_test, batch_size=16)
print(y)
