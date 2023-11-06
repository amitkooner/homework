#!/usr/bin/env python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pickle
from flask import Flask, request, jsonify
import numpy as np

# Load the dataset
spotify_data = pd.read_csv('/Users/AKooner/Desktop/coding/homework/mlzoomcamp_midterm/spotify_songs.csv')

# Data preprocessing
# Transforming 'track_album_release_date' into datetime and calculating the song age in days
spotify_data['track_album_release_date'] = pd.to_datetime(spotify_data['track_album_release_date'])
reference_date = pd.to_datetime('2023-04-30')  # Assuming this is the date of data collection
spotify_data['song_age_days'] = (reference_date - spotify_data['track_album_release_date']).dt.days

# Features to include in the model
features_to_include = [
    'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 
    'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 
    'duration_ms', 'playlist_genre', 'playlist_subgenre', 'song_age_days'
]

# Target variable
target = 'track_popularity'

# Separating features and target variable
X = spotify_data[features_to_include]
y = spotify_data[target]

# Data splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Defining numerical and categorical features
numeric_features = [
    'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 
    'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 
    'duration_ms', 'song_age_days'
]
categorical_features = ['playlist_genre', 'playlist_subgenre']

# Creating transformers for numeric and categorical data
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Combining transformers into a preprocessor step
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Applying the preprocessing pipeline to the training data
X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)

# Initialize the Random Forest model
rf_model = RandomForestRegressor(random_state=42)

# Train the model on the entire training set
rf_model.fit(X_train_preprocessed, y_train)

# Model evaluation
y_train_pred = rf_model.predict(X_train_preprocessed)
y_test_pred = rf_model.predict(X_test_preprocessed)
train_rmse = mean_squared_error(y_train, y_train_pred, squared=False)
test_rmse = mean_squared_error(y_test, y_test_pred, squared=False)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

# Output the performance metrics
print(f'Train RMSE: {train_rmse}')
print(f'Test RMSE: {test_rmse}')
print(f'Train R^2: {train_r2}')
print(f'Test R^2: {test_r2}')

# Save the model to a file using pickle
model_filename = 'rf_model.pkl'
with open(model_filename, 'wb') as file:
    pickle.dump(rf_model, file)

# Flask web service
app = Flask(__name__)

# Load the trained model
with open(model_filename, 'rb') as file:
    model = pickle.load(file)

# Define endpoint for predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Get the features from the POST request
    data = request.get_json(force=True)
    
    # Ensure that we received the correct number of features
    # This should match the number of features the model expects
    if len(data['features']) == len(features_to_include) - 2: # excluding categorical features
        # Preprocess the incoming features
        features = np.array(data['features']).reshape(1, -1)
        features_preprocessed = preprocessor.transform(features)
        
        # Make prediction
        prediction = model.predict(features_preprocessed)
        
        # Send back the prediction as a json response
        return jsonify(prediction.tolist())
    else:
        return jsonify({"error": "Invalid number of features"}), 400

if __name__ == '__main__':
    app.run(debug=True)