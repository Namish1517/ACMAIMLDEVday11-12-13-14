import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,classification_report,roc_curve,roc_auc_score

data=load_breast_cancer()
X,y=data.data,data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
models = {"Logistic Regression":LogisticRegression(max_iter=1000),"Decision Tree":DecisionTreeClassifier(random_state=42),"Random Forest":RandomForestClassifier(random_state=42),"SVM (RBF)":SVC(kernel='rbf',probability=True)}
print("Cross-Validation Accuracy")
for name,model in models.items():

    scores=cross_val_score(model,X,y,cv=5,scoring='accuracy')
    print(f"{name}:Mean Accuracy={scores.mean():.4f}")
plt.figure()
for name,model in models.items():
    model.fit(X_train,y_train)
    y_prob=model.predict_proba(X_test)[:,1]
    fpr,tpr,_=roc_curve(y_test,y_prob)
    auc_score=roc_auc_score(y_test,y_prob)
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc_score:.2f})")
plt.plot([0,1],[0,1],'k')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves")
plt.legend()

plt.show()
param_grid={'n_estimators':[50,100, 200],'max_depth':[None,5,10],'min_samples_split':[2,5,10]}
grid=GridSearchCV(RandomForestClassifier(random_state=42),param_grid,cv=5,scoring='accuracy')
grid.fit(X_train,y_train)
print("Random Forest GridSearchCV")
print("Best Params:",grid.best_params_)
print("Best CV Score:",grid.best_score_)
best_rf=grid.best_estimator_
y_pred = best_rf.predict(X_test)
print("\nBest Random Forest on Test Set")
print("Accuracy:",accuracy_score(y_test,y_pred))
print("Classification Report:",classification_report(y_test,y_pred))
