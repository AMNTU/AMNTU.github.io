from sklearn.preprocessing import LabelEncoder
from flask import Flask, request, render_template, render_template_string
from pyngrok import ngrok
from pytorch_tabnet.tab_model import TabNetRegressor
import xgboost as xgb
import pandas as pd
import pandas.api.types as ptypes
import numpy as np
import threading
import datetime
import threading

df_hdbInfo = pd.read_csv('HDBInfo.csv')     # Load the master dataset for lookup of HDB block parameters
df_prepared = pd.read_csv('Prepared.csv')   # Load the transaction dataset for lookup of town maturity and most recent price sqm

# Cast 'month' as datetime type, if not, before extracting 'years' from datetime to numerical
if not ptypes.is_datetime64_any_dtype(df_prepared['month']):
  df_prepared['month'] = pd.to_datetime(df_prepared['month'], utc=True).dt.tz_localize(None)
df_prepared['years'] = df_prepared['month'].dt.year

# Create a feature that counts the number of months from baseline of the earliest month (Jan 2017), as the models cannot accept datetime data type
df_prepared['months'] = ((df_prepared['month'].dt.year - df_prepared['month'].min().year) * 12 + (df_prepared['month'].dt.month - df_prepared['month'].min().month))

# Drop the original datetime column since we already have the count of number of months
df_prepared = df_prepared.drop(columns=['month'], axis=1)

# Convert distance from km to m then convert all features to 'int32' data type for easier working with numpy array later
df_prepared['distance_km'] = df_prepared['distance_km'] * 1000
df_prepared = df_prepared.astype('int32')

# Load the encoder for mapping the towns' string back to the same integer used during training
town_mapping = pd.read_csv('town_encoder.csv')
town_encoder = LabelEncoder()                             # Rebuild the LabelEncoder from the loaded mapping
town_encoder.classes_ = town_mapping['original'].values   # Set the classes manually

df_prepared.info(), df_prepared.head()

# # Load the trained XGBoost model
xgb_model = xgb.XGBRegressor()
xgb_model.load_model('xgboost.json')

# Load the trained TabNet model
# tabnet_model = TabNetRegressor()
# tabnet_model.load_model('tabnet_model.zip')

# Look up df_prepared transaction dataset to find the price_sqm of the latest transaction that satisfies the conditions
def get_price_sqm(town, distance, level, sqm):
    upper, lower = sqm + 3, sqm - 3                   # Define the upper and lower range for the floor area sqm
    for index in df_prepared.index[::-1]:
      row = df_prepared.loc[index]
      if( row['town'] == town and
          row['distance_km'] == distance and
          row['level'] == level and
          lower <= row['floor_area_sqm'] <= upper):   # Check if floor area is within a range instead of fixed integer to allow for slight input error
          price_sqm = row['price_sqm']
          print(f"Found matching row with price_sqm: {price_sqm}")
          break
    return price_sqm

# Look up df_hdbInfo master dataset with postal code
def lookup_values(features, postal_code):

    # Return town, remaining lease and distance km from lookup
    try:
      town_name = df_hdbInfo.loc[df_hdbInfo['postalcode'] == postal_code, 'towns'].iloc[0]                  # Lookup the town name from master dataset
      features[0] = town_encoder.transform([town_name])[0]                                                  # Retrieve the encoded town number
      year_completed = df_hdbInfo.loc[df_hdbInfo['postalcode'] == postal_code, 'year_completed'].iloc[0]    # Retrieve the year completed to calculate remaining lease
      features[2] = (99 - (datetime.datetime.now().year - year_completed)) * 12                             # Calculate remaining lease in months
      features[3] = df_prepared.loc[df_prepared['town'] == features[0], 'mature'].iloc[0]                   # Retrieve town maturity
      features[6] = (df_hdbInfo.loc[df_hdbInfo['postalcode'] == postal_code, 'distance_km'].iloc[0]) * 1000 # Retrieve the distance to MRT
      features[5] = get_price_sqm(features[0], features[6], features[4], features[1])                       # Retrieve the price per sqm
      features[7] = datetime.datetime.now().year                                                            # Retrieve the current year
      features[8] = ((features[7] - 2017) * 12) + datetime.datetime.now().month                             # Retrieve the number of months since baseline Jan 2017

      return features

    except KeyError:
      print(f"Postal code '{postal_code}' not found.")
      return None

# Define a function to make predictions by ensembling (average) the predictions from XGBoost and TabNet models
def predict_house_price(features, postal_code):
      features = lookup_values(features, postal_code)
      features = np.array(features).reshape(1, -1)      # Reshape for single prediction
      xgb_prediction = xgb_model.predict(features)
    #   tabnet_prediction = tabnet_model.predict(features)
      prediction = xgb_prediction
    #   prediction = (xgb_prediction + tabnet_prediction)/2
      print(features)
    #   return prediction[0][0].astype('int32')
      return prediction[0][0].astype('int32')

features = [0, 65, 0, 0, 2, 0, 0, 0, 0]            # Initialise the array
print(predict_house_price(features, "120703"))     # Send parameters to function for prediction


# # Initialize Flask app
# app = Flask(__name__)
# port = "5000"

# # Set up NGROK reverse proxy to allow access when running from online notebooks such as Google Colab
# public_url = ngrok.connect(5000).public_url # Expose port 5000 (Flask's default)
# print(f" * ngrok tunnel \"{public_url}\" -> \"http://127.0.0.1:{port}\"")
# app.config["BASE_URL"] = public_url         # Update any base URLs to use the public ngrok URL

# # Define the route for the homepage
# @app.route('/')
# def home():
#     # return render_template('index.html')  # Use render_template
#     return render_template_string("""
#         <!DOCTYPE html>
#         <html lang="en">
#         <head>
#             <meta charset="UTF-8">
#             <meta name="viewport" content="width=device-width, initial-scale=1.0">
#             <title>HDB Resale Housing Price Guidance</title>
#         </head>
#         <body>
#             <h1>HDB Resale Housing Price Guidance</h1>
#             <p>This app provides a guide for the offering price of a HDB resale flat. The aim is to help buyers make a reasonable offer price for the flat of their choice. </p>
#             <form action="/predict" method="POST">
#                 <label for="postcode">Postal Code:</label><br>
#                 <input type="text" id="postcode" name="postcode"><br><br>
#                 <label for="sqm">Floor Area (in sqm):</label><br>
#                 <input type="text" id="sqm" name="sqm"><br><br>
#                 <label for="level">Floor Level:</label><br>
#                 <select id="level" name="level">
#                   <option value="1">Low</option>
#                   <option value="2">Mid</option>
#                   <option value="3">High</option>
#                 </select><br><br>
#                 <input type="submit" value="Submit">
#             </form>
#         </body>
#         </html>
#     """)

# # Define the route to handle form submission
# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         # Get input values from the form
#         param1 = str(request.form['postcode'])
#         param2 = int(request.form['sqm'])
#         param3 = int(request.form['level'])

#         # Make the prediction using the model
#         features = [0, param2, 0, 0, param3, 0, 0, 0, 0]            # Initialise the array
#         predicted_price = predict_house_price(features, param1)     # Send parameters to function for prediction

#         # Return the result to the user
#         # return render_template('result.html', price=predicted_price)
#         return render_template_string("""
#             <!DOCTYPE html>
#             <html lang="en">
#             <head>
#                 <meta charset="UTF-8">
#                 <meta name="viewport" content="width=device-width, initial-scale=1.0">
#                 <title>Transaction Price Guidance</title>
#             </head>
#             <body>
#                 <h1>A reasonable transaction price will be S${{ price }}</h1>
#                 <br>
#                 <a href="/">Go back</a>
#             </body>
#             </html>
#         """, price = predicted_price)

#     except Exception as e:
#         return f"Error: {str(e)}"

# # Start the Flask server in a new thread
# threading.Thread(target=app.run, kwargs={"use_reloader": False}).start()