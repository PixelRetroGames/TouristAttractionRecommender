import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Dense, Concatenate
from tensorflow.keras.models import Model

df = pd.read_csv("small.df")

print(df)
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

# Function to perform partial fit
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

# New data
# for i in range(100):
#     new_data = pd.DataFrame({'user_name': ["Rob Polley"], 'rating': [5], 'location_name': ["McDonald's"]})
#     partial_fit(model, new_data)

print(user_name_mapping)
user_name_to_recommend = 'Rob Polley'  # Replace with the actual user name
recommended_locations = get_recommendation(model, user_name_to_recommend, num_locations, user_name_mapping, location_mapping)

print("Recommended locations:", recommended_locations)
