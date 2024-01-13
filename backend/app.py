from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Dense, Concatenate
from tensorflow.keras.models import Model

model = None
user_name_mapping = None
location_mapping = None
num_users = None
num_locations = None
current_user = None

def build_recommendation_model(num_users, num_locations, embedding_dim=50):
    user_input = Input(shape=(1,), name='user_input')
    location_input = Input(shape=(1,), name='location_input')

    user_embedding = Embedding(input_dim=num_users, output_dim=embedding_dim)(user_input)
    location_embedding = Embedding(input_dim=num_locations, output_dim=embedding_dim)(location_input)

    user_flat = tf.keras.layers.Flatten()(user_embedding)
    location_flat = tf.keras.layers.Flatten()(location_embedding)

    concatenated = Concatenate()([user_flat, location_flat])

    dense1 = Dense(128, activation='relu')(concatenated)
    dense2 = Dense(64, activation='relu')(dense1)

    output = Dense(1, activation='sigmoid')(dense2)

    model = Model(inputs=[user_input, location_input], outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def partial_fit(model, new_data):
    new_user_ids = new_data['user_name'].astype("category").cat.codes.values
    new_location_ids = new_data['location_name'].astype("category").cat.codes.values
    new_y = new_data['rating']

    for i in range(len(new_user_ids)):
        model.train_on_batch({'user_input': np.array([new_user_ids[i]]), 'location_input': np.array([new_location_ids[i]])}, np.array([new_y[i]]))

def get_recommendation(model, user_name, num_locations, user_name_mapping, location_mapping):
    # Get the user ID using the provided mapping
    user_id = user_name_mapping[user_name]

    user_array = np.full(num_locations, user_id)
    location_array = np.arange(num_locations)

    user_input = {'user_input': user_array, 'location_input': location_array}

    predictions = model.predict(user_input)
    print("Predictions: " +str(predictions))
    # Sort locations based on predicted ratings
    sorted_locations = np.argsort(predictions.flatten())[::-1]

    # Map location IDs back to location names
    recommended_locations = [location_mapping[loc_id] for loc_id in sorted_locations]

    return recommended_locations

# # Simulated user data (replace with your user database logic)
users = {
    'Roger Kaid': {'password': 'pwd'},
    'Rob Polley': {'password': 'pwd'}
}

def compute_recommendation(user):
    global num_locations, user_name_mapping, location_mapping
    print(user_name_mapping)
    return get_recommendation(model, user, num_locations, user_name_mapping, location_mapping)[0]

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route("/login", methods=["POST"])
def login():
    data = request.get_json()
    username = data.get('username')
    global current_user
    current_user = username

    credentials = users.get(username)
    if credentials:
        return jsonify({'message': 'Login Successful'}), 200
    else:
        return jsonify({'message': 'Invalid credentials'}), 401

@app.route("/locations", methods=["GET"])
def locations():
    return jsonify(df['location_name'].unique().tolist()), 200

@app.route("/reviews", methods=["POST"])
def reviews():
    data = request.get_json()
    username = data.get('username')
    location = data.get('location')
    rating = data.get('rating')
    print(f"username: {username}, location: {location}, rating: {rating}")
    global model
    partial_fit(model, pd.DataFrame({'user_name': [username], 'rating': [int(rating)], 'location_name': [location]}))
    return jsonify({'message': 'Inserted data successfully'}), 200

@app.route("/recommendation", methods=["GET"])
def recommendation():
    global current_user
    return jsonify(compute_recommendation(current_user)), 200

@app.route("/logout", methods=["GET"])
def logout():
    global current_user
    current_user = None
    return jsonify("Successful logout"), 200

if __name__ == '__main__':
    df = pd.read_csv("../small.df")
    # Assume df is your DataFrame with columns: user, rating, location_name
    num_users = len(df['user_name'].unique())
    num_locations = len(df['location_name'].unique())

    # Create mappings for user and location names to IDs
    user_name_mapping = dict(zip(df['user_name'].unique(), range(num_users)))
    location_mapping = dict(zip(range(num_locations), df['location_name'].unique()))

    model = build_recommendation_model(num_users, num_locations)

    # Initial training
    user_ids = df['user_name'].astype("category").cat.codes.values
    location_ids = df['location_name'].astype("category").cat.codes.values
    y = df['rating']

    model.fit({'user_input': user_ids, 'location_input': location_ids}, y, epochs=10, batch_size=64)
    app.run(debug=True)