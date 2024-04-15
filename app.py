import pandas as pd
import numpy as pn

from flask import Flask, request, render_template

from sklearn.preprocessing import StandardScaler

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

# Entry point for executing app
app=Flask(__name__)

# Route for home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(
            Bedrooms=int(request.form.get('Bedrooms')),
            Bathrooms=int(request.form.get('Bathrooms')),
            Parking_spaces=int(request.form.get('Parking_spaces')),
            Suburb=request.form.get('Suburb'),
            )

        pred_df=data.get_data_as_data_frame()

        predict_pipeline=PredictPipeline()
        output_result=predict_pipeline.predict(pred_df)
        results = round(output_result[0] / 100) * 100
        return render_template('home.html', results=results)

if __name__=="__main__":
    app.run(host="0.0.0.0", debug=True)