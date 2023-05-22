from flask import Flask , request, app, render_template
from  flask import Response
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
app = Flask(__name__)

model=pickle.load(open('/config/workspace/Model/Model_for_predication.pkl', 'rb'))
scaler=pickle.load(open('/config/workspace/Model/StandardScaler.pkl', 'rb'))

## route for homepage
@app.route("/")
def index():
    return render_template('index.html')

## Route for single data point predication

@app.route('/predictdata', methods=['GET','POST'])
def predict_datapoint():

    if request.method=='POST':
        Pregnancies= (request.form.get('Pregnancies'))
    
        Glucose=float(request.form.get('Glucose'))
        BloodPressure=float(request.form.get('BloodPressure'))
        SkinThickness=float(request.form.get('SkinThickness'))
        Insulin=float(request.form.get('Insulin'))
        BMI=float(request.form.get('BMI'))
        DiabetesPedigreeFunction=float(request.form.get('DiabetesPedigreeFunction'))
        Age=float(request.form.get('Age'))
        X = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]
        data=[float(i) for i in X]
        final_input=np.array(data).reshape(1,-1)
        print(final_input)   
    
        new_data=scaler.transform(final_input)
        predict=model.predict(new_data)
        if predict[0]==1:
            result='Diabetic'
        else:
            result='Non-Diabetic'

        return render_template('single_prediction.html', result=result)

    else:
        return render_template('home.html')
    
if __name__=="__main__":
    app.run(host="0.0.0.0")
