import numpy as np
import pandas as pd
import tensorflow as tf
from flask import Flask, request, render_template
from sklearn import preprocessing
import pickle

app = Flask(__name__)

model = pickle.load(open('/Users/shubhan/Desktop/Projects/HousingPricesEstimator/models/model.pk1', 'rb'))

# Adapt the city lookup 
data = pd.read_csv('/Users/shubhan/Desktop/California_Real_Estate.csv')
data = data.dropna(axis = 0)
unique_cities = data['city'].unique()
cities = pd.DataFrame(unique_cities, columns = ['city'])
city_lookup = tf.keras.layers.StringLookup()
city_lookup.adapt(cities['city'])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods = ['POST'])
def predict():
    values = list(request.form.values())
    int_features = [float(value) for value in values[:5]]

    city = values[5]
    city_encoded = city_lookup(city).numpy()
    int_features.append(float(city_encoded))  
    int_features.append(float(values[6]))

    features = np.array([int_features])

    scaled_inputs = preprocessing.scale(features)

    prediction = model.predict(scaled_inputs)

    predicted_value = prediction[0][0]

    output = round(predicted_value, 2)

    return render_template('index.html', prediction_text = 'House Price: {}'.format(output))

if __name__ == "__main__":
    app.run()
