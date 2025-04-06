from flask import Flask, request, render_template_string
from pyngrok import ngrok
import os
from sklearn.preprocessing import LabelEncoder
# from pytorch_tabnet.tab_model import TabNetRegressor  (this works but not in MacOS due to Segmentation fault 11)
# import xgboost as xgb                                 (this works but not in MacOS due to Segmentation fault 11)
import pandas as pd
import pandas.api.types as ptypes
import numpy as np
import datetime
import multiprocessing

# # Load the trained XGBoost model  (this works but not in MacOS due to Segmentation fault 11)
# xgb_model = xgb.XGBRegressor()
# xgb_model.load_model('xgboost.json')

# # Load the trained TabNet model   (this works but not in MacOS due to Segmentation fault 11)
# tabnet_model = TabNetRegressor()
# tabnet_model.load_model('tabnet_model.zip')

# # Look up df_prepared transaction dataset to find the price_sqm of the latest transaction that satisfies the conditions
# def get_price_sqm(town, distance, level, sqm, lease):
#     upper, lower = sqm + 3, sqm - 3                     # Define the upper and lower range for the floor area sqm
#     print(town, distance, level, sqm, lease)
#     for index in df_prepared.index[::-1]:
#       row = df_prepared.loc[index]
#       if( row['town'] == town and
#           row['distance_km'] == (distance) and
#           row['level'] == level and
#           lower <= row['floor_area_sqm'] <= upper and   # Check if floor area is within a range instead of fixed integer to allow for slight input error
#           row['remaining_lease'] >= lease):             # Check if remaining lease is the same or greater
#           price_sqm = row['price_sqm']
#           print(f"Found matching row with price_sqm: {price_sqm}")
#           break
#     return price_sqm

def get_price_sqm(town, distance, level, sqm, lease):
    upper, lower = sqm + 3, sqm - 3                         # Define the upper and lower range for the floor area sqm
    print(town, distance, level, sqm, lease)
    for index in df_prepared.index[::-1]:
      row = df_prepared.loc[index]
      if( row['town'] == town and
          row['distance_km'] == distance and
          row['level'] == level and
          lower <= row['floor_area_sqm'] <= upper and       # Check if floor area is within a range instead of fixed integer to allow for slight input error
          lease <= row['remaining_lease'] <= (lease + 36)): # Check if remaining lease matches within last 3 years
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
      features[5] = get_price_sqm(features[0], features[6], features[4], features[1], features[2])                       # Retrieve the price per sqm
      features[7] = datetime.datetime.now().year                                                            # Retrieve the current year
      features[8] = ((features[7] - 2017) * 12) + datetime.datetime.now().month                             # Retrieve the number of months since baseline Jan 2017

      return features

    except KeyError:
      print(f"Postal code '{postal_code}' not found.")
      return None

def run_xgboost(features, queue):
    import xgboost as xgb
    xgb_model = xgb.XGBRegressor()
    xgb_model.load_model('xgboost.json')
    features = np.array(features).reshape(1, -1)      # Reshape for single prediction
    prediction = xgb_model.predict(features)
    queue.put(int(prediction))  # Put the result in the queue
    # print(f"XGBoost prediction: {prediction}")

def run_tabnet(features, queue):
    from pytorch_tabnet.tab_model import TabNetRegressor
    tabnet_model = TabNetRegressor()
    tabnet_model.load_model('tabnet_model.zip')
    features = np.array(features).reshape(1, -1)      # Reshape for single prediction
    prediction = tabnet_model.predict(features)
    queue.put(int(prediction[0]))  # Put the result in the queue
    # print(f"TabNet prediction: {prediction[0]}")

# Define a function to make predictions by ensembling (average) the predictions from XGBoost and TabNet models
def predict_house_price(features, postal_code):
    features = lookup_values(features, postal_code)
    features = np.array(features).reshape(1, -1)      # Reshape for single prediction

    # Create queues to collect results
    xgboost_queue = multiprocessing.Queue()
    tabnet_queue = multiprocessing.Queue()

    # For MacOS, run in separate multiprocessing to avoid Segmentation fault 11
    xgboost_process = multiprocessing.Process(target=run_xgboost, args=(features, xgboost_queue))
    tabnet_process = multiprocessing.Process(target=run_tabnet, args=(features, tabnet_queue))

    xgboost_process.start()
    tabnet_process.start()

    xgboost_process.join()
    tabnet_process.join()

    # Retrieve results from queues
    xgboost_result = xgboost_queue.get()
    tabnet_result = tabnet_queue.get()

    print(f"S${xgboost_result}, S${tabnet_result}")
    return int((xgboost_result + tabnet_result) / 2)

    # xgb_prediction = xgb_model.predict(features)          (this works but not in MacOS due to Segmentation fault 11)
    # tabnet_prediction = tabnet_model.predict(features)    (this works but not in MacOS due to Segmentation fault 11)
    # prediction = (xgb_prediction + tabnet_prediction)/2   (this works but not in MacOS due to Segmentation fault 11)
    # print(features)
    # return prediction[0].astype('int32')

if __name__ == "__main__":
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

    # # For troubleshooting to ensure the models are working
    # features = [9, 65, 734, 1, 2, 6000, 500, 2025, 100]       # Initialise the array
    # print(predict_house_price(features, "120703"))            # Send parameters to function for prediction

    # Initialize Flask app to serve the web frontend
    app = Flask(__name__)
    port = "5000"

    # Set up NGROK reverse proxy to allow access when running from online notebooks such as Google Colab
    public_url = ngrok.connect(5000).public_url # Expose port 5000 (Flask's default)
    print(f" * ngrok tunnel \"{public_url}\" -> \"http://127.0.0.1:{port}\"")
    app.config["BASE_URL"] = public_url         # Update any base URLs to use the public ngrok URL

    # Define the route for the homepage
    @app.route('/')
    def home():
        # Read the index.html file from the root folder instead of /templates folder (default of flask)
        with open(os.path.join(os.getcwd(), 'index.html'), 'r') as file:
            html_content = file.read()
        return render_template_string(html_content)

    # Define the route to handle form submission
    @app.route('/predict', methods=['POST'])
    def predict():
        try:
            # Get input values from the form
            param1 = str(request.form['postcode'])
            param2 = int(request.form['sqm'])
            param3 = int(request.form['level'])

            # Make the prediction using the model
            features = [0, param2, 0, 0, param3, 0, 0, 0, 0]                    # Initialise the array
            predicted_price = predict_house_price(features, param1)             # Send parameters to function for prediction

            # Read the index.html file from the root folder instead of /templates folder (default of flask)
            with open(os.path.join(os.getcwd(), 'result.html'), 'r') as file:
                html_content = file.read()
            return render_template_string(html_content, price=predicted_price)  # Return the result to the user

        except Exception as e:
            return f"Error: {str(e)}"
    
    # Start the Flask server in the main thread
    app.run(port=5000, use_reloader=False)