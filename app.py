# Import required libraries
from utils.model import *
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np

# Data location
DATA_PATH = "../data/"
# Encoders location
ENCODERS_PATH = "encoders/"
# Model location
MODEL_PATH = "models/"



app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    # Get locations from the csv file
    df = pd.read_csv(DATA_PATH + 'transformed_dataset.csv')
    locations = np.sort(df['location'].unique()).tolist()
    remove_locations = ['Banbury', 'Billingshurst', 'Coatbridge', 'Dalkeith', 'Dunbar', 'Exeter', \
                        'Haddington', 'Huntly', 'Leamington-Spa', 'Nantwich', 'Newbury', 'North Berwick', 'North Shields',\
                            'On-fleet Bay', 'Poole', 'Putney', 'South Shields', 'Sunderland', 'Upper Tooting', 'Wandsworth', \
                                'Warwick', 'Wokingham', 'Worthing']
    for location in remove_locations:
        locations.remove(location)

    if request.method == 'POST':
        location = request.form['location']
        hour_of_the_day = float(request.form['hour_of_the_day'])

        # Load historical data
        scaled_df, df = load_preprocessed_data()
        predictions_df = predict_rates(scaled_df, location)
        demand_factor_value, peak_hours = demand_factor(df, location, hour_of_the_day)

        predictions_df = apply_dynamic_pricing_strategy(predictions_df, demand_factor_value)

        # adjust final prices
        grouped_averages = get_average_rates(df, location)
        predictions_df = predictions_df.apply(lambda row: adjust_prices(row, grouped_averages), axis=1)
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

        # Initialize sums
        total_adjusted_revenue = 0
        total_actual_revenue = 0
        total_profitability = 0

        # Iterate through the dictionary and sum up the values
        for key, value in profitability.items():
            total_adjusted_revenue += value['adjusted_revenue']
            total_actual_revenue += value['actual_revenue']
            total_profitability += value['profitability']

        # Add the new key with summed values
        profitability['Z'] = {
            'adjusted_revenue': total_adjusted_revenue,
            'actual_revenue': total_actual_revenue,
            'profitability': total_profitability
        }

        return jsonify(peakHours=peak_hours, predictions=predictions, profitability=profitability)
    return render_template('index.html', locations=locations)

if __name__ == '__main__':
    app.run()
