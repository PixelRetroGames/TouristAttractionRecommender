import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split

# Assuming the user_model, location_model, TwoTowerModel, etc., are defined as in your previous code

# Load the data
ratings = pd.read_csv("/content/drive/MyDrive/fac/master/sadc/dataset/dataset.csv")

# Map user and location IDs to integer indices
user_mapping = {user: idx for idx, user in enumerate(ratings['user'].unique())}
location_mapping = {location: idx for idx, location in enumerate(ratings['gmap_id'].unique())}

ratings['user_index'] = ratings['user'].map(user_mapping)
ratings['location_index'] = ratings['gmap_id'].map(location_mapping)

# Split the data into training and testing sets
train_data, test_data = train_test_split(ratings, test_size=0.2, random_state=42)

# Define embedding dimensions
embedding_dimension = 32

# User model
user_model = tf.keras.Sequential([
    tf.keras.layers.StringLookup(
        vocabulary=list(user_mapping.keys()), mask_token=None),
    tf.keras.layers.Embedding(len(user_mapping) + 1, embedding_dimension)
])

# Location model
location_model = tf.keras.Sequential([
    tf.keras.layers.StringLookup(
        vocabulary=list(location_mapping.keys()), mask_token=None),
    tf.keras.layers.Embedding(len(location_mapping) + 1, embedding_dimension)
])

# Two-tower model
class TwoTowerModel(tf.keras.Model):
    def _init_(self, user_model, location_model):
        super(TwoTowerModel, self)._init_()
        self.user_model = user_model
        self.location_model = location_model

    def call(self, inputs):
        user_embeddings = self.user_model(inputs['user'])
        location_embeddings = self.location_model(inputs['gmap_id'])
        return user_embeddings, location_embeddings

# Instantiate the two-tower model
model = TwoTowerModel(user_model, location_model)

# Define loss and metrics
loss = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adagrad(learning_rate=0.1)
metrics = [tf.keras.metrics.RootMeanSquaredError()]

# Compile the model
model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

# Load the pretrained weights
checkpoint_path = 'drive/My Drive/fac/master/sadc/trained_model/'
model.load_weights(checkpoint_path)