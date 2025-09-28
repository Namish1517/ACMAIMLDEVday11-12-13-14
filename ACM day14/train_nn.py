import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

data=load_breast_cancer()

X,y=data.data,data.target
scaler=StandardScaler()
X=scaler.fit_transform(X)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)

model=Sequential([Dense(16,activation='relu',input_shape=(X_train.shape[1],)),Dense(8,activation='relu'),Dense(1,activation='sigmoid')])
model.compile(optimizer=Adam(learning_rate=0.01),loss='binary_crossentropy',metrics=['accuracy'])
model.fit(X_train,y_train,epochs=20,batch_size=16,verbose=1)

loss,acc=model.evaluate(X_test,y_test,verbose=0)
print("Test Accuracy:",acc)

model.save("models/nn_model.h5")
print("Neural Network trained and saved.")
