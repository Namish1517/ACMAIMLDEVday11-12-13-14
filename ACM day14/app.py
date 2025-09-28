from flask import Flask,request,jsonify
import joblib
import numpy as np
app=Flask(__name__)
model=joblib.load('models/voting_model.pkl')
@app.route('/predict',methods=['POST'])
def predict():
    data=request.json
    features=np.array(data['features']).reshape(1,-1)
    prediction=model.predict(features)[0]
    proba=model.predict_proba(features)[0][1]
    return jsonify({"prediction":int(prediction),"probability":float(proba)})
if __name__=='__main__':
    app.run(debug=True)
