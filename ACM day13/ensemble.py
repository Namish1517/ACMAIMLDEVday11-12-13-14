from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score,classification_report

data=load_breast_cancer()
X ,y=data.data,data.target
X_train,X_test,y_train,y_test=train_test_split(X ,y,test_size=0.3 ,random_state=42)
xgb=XGBClassifier(use_label_encoder=False,eval_metric='logloss',random_state=42)
xgb.fit(X_train ,y_train)
y_pred_xgb=xgb.predict(X_test)
print("XGBoost Accuracy:",accuracy_score(y_test,y_pred_xgb))
print("XGBoost Report:\n",classification_report(y_test,y_pred_xgb))
vc=VotingClassifier(estimators=[('lr',LogisticRegression( max_iter=1000)),('dt',DecisionTreeClassifier(random_state=42)),('rf',RandomForestClassifier(random_state=42))],voting='soft')
vc.fit(X_train,y_train)
y_pred_vc=vc.predict(X_test)
print("Voting Classifier Accuracy:",accuracy_score(y_test,y_pred_vc))
print("Voting Classifier Report:\n",classification_report(y_test,y_pred_vc))
'''
XGBoost Accuracy: 0.9649122807017544
XGBoost Report:
               precision    recall  f1-score   support

           0       0.94      0.97      0.95        63
           1       0.98      0.96      0.97       108

    accuracy                           0.96       171
   macro avg       0.96      0.97      0.96       171
weighted avg       0.97      0.96      0.97       171
Voting Classifier Accuracy: 0.9824561403508771
Voting Classifier Report:
               precision    recall  f1-score   support

           0       0.98      0.97      0.98        63
           1       0.98      0.99      0.99       108

    accuracy                           0.98       171
   macro avg       0.98      0.98      0.98       171
weighted avg       0.98      0.98      0.98       171

'''

