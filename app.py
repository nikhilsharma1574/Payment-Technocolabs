import json
import pickle

from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd

app=Flask(__name__)
## Load the mod
model1=pickle.load(open('Mortgage_delinquency_final.pkl','rb'))
model2=pickle.load(open('Prepayment_Risk_final.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')



@app.route('/predict',methods=['POST','GET'])
def predict():
    int_features=[int(x) for x in request.form.values()]
    final=[np.array(int_features)]
    print(int_features)
    print(final)
    prediction=model1.predict(final)
    prediction1=model2.predict(final)



    if (prediction==1):
            return render_template( "home.html", prediction_text=f"Loan is non Delinquent and The prepayment risk is   {prediction1} %")

    else: 
            return render_template('home.html', prediction_text='Loan is Delinquent No chance of prepayment')
    



    
if __name__ == '__main__':
    app.run(debug=True)


