from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import joblib

data=load_breast_cancer()
X,y=data.data,data.target
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)
vc=VotingClassifier(estimators=[('lr',LogisticRegression(max_iter=1000)),('dt',DecisionTreeClassifier(random_state=42)),('rf',RandomForestClassifier(random_state=42))],voting='soft')
vc.fit(X_train,y_train)
joblib.dump(vc,'models/voting_model.pkl')
print("Voting Classifier trained and saved.")
