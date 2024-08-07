# Import required libraries
from utils.model import *
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import warnings
from scipy import stats
from matplotlib.ticker import FuncFormatter
from category_encoders import BinaryEncoder, OneHotEncoder
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
import joblib

# Data location
DATA_PATH = "../data/"
# Encoders location
ENCODERS_PATH = "encoders/"
# Model location
MODEL_PATH = "models/"

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    df = pd.read_csv(DATA_PATH + 'transformed_dataset.csv')
    locations = np.sort(df['location'].unique())

    # locations_list = []
    # locations = []
    # with open("locations.txt", "r") as f:
    #     for location in f.readlines():
    #         locations_list.append(location)
    # locations_list.remove('\n')
    # for location in locations_list:
    #     locations.append(location.strip())
    # print(locations, len(locations))

    if request.method == 'POST':
        location = request.form['location']
        print(location, '\n\n')
        hour_of_the_day = float(request.form['hour_of_the_day'])

        # Load historical data
        scaled_df, df = load_preprocessed_data()
        predictions_df = predict_rates(scaled_df, location)
        demand_factor_value, peak_hours = demand_factor(df, location, hour_of_the_day)

        predictions_df = apply_dynamic_pricing_strategy(predictions_df, demand_factor_value)

        # adjust final prices
        predictions_df = predictions_df.apply(adjust_prices, axis=1)
        predictions_df.rename(columns={'hourly_rate': 'current_hourly_rate', 
                                       'daily_rate': 'current_daily_rate', 'final_hourly_rate': 'adjusted_hourly_rate',
                                       'final_daily_rate': 'adjusted_daily_rate'}, inplace=True)
        predictions = dict()

        # store all values of predictions_df in predictions dict
        for index, row in predictions_df.iterrows():
            predictions[row['vehicle_type']] = {
                'current_hourly_rate': row['current_hourly_rate'],
                'current_daily_rate': row['current_daily_rate'],
                'adjusted_hourly_rate': row['adjusted_hourly_rate'],
                'adjusted_daily_rate': row['adjusted_daily_rate']
            }

        # Predict demand_factor for Profitability Calculation
        df_merged_revenue = calculate_profitability(df, location, predictions_df)

        profitability = dict()

        # store all values of predictions_df in predictions dict
        for index, row in df_merged_revenue.iterrows():
            profitability[row['vehicle_type']] = {
                'adjusted_revenue': row['adjusted_revenue'],
                'actual_revenue': row['actual_revenue'],
                'profitability': row['profitability']
            }

        return jsonify(peakHours=peak_hours, predictions=predictions, profitability=profitability)
    return render_template('index.html', locations=locations)

if __name__ == '__main__':
    app.run(debug=True)
