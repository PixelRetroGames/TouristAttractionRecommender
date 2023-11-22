import csv
import gzip
import json
import os
import random
from typing import Dict, Text

import numpy as np
import pandas as pd
import tensorflow as tf

import tensorflow_datasets as tfds
import tensorflow_recommenders as tfrs
from tensorflow.python.client import device_lib


def parseCSVGZIP(path):
    csvf = gzip.open(path, 'rt')
    return parseCSVFile(csvf)


def parseCSV(path):
    csvf = open(path, 'rt')
    return parseCSVFile(csvf)


def parseCSVFile(file):
    csvReader = csv.DictReader(file)
    for rows in csvReader:
        yield rows


def writeCSVGZIP(data, path, filter=None):
    csvf = gzip.open(path, "wt", newline='')
    return writeCSVFile(data, csvf, filter)


def writeCSV(data, path, filter=None):
    csvf = open(path, "w", newline='')
    return writeCSVFile(data, csvf, filter)


def writeCSVFile(data, file, filter=None):
    if filter is None:
        filter = {}

    csvWriter = None
    for elem in data:
        if csvWriter == None:
            csvWriter = csv.DictWriter(file, elem.keys())
            csvWriter.writeheader()

        skip = False
        for key in filter:
            if elem[key] != filter[key]:
                skip = True

        if not skip:
            csvWriter.writerow(elem)

    file.close()


def labelTestAndTrain(source, testProb):
    for element in source:
        element["Usage"] = random.choices(["Test", "Train"], [testProb, 1 - testProb])[0]
        yield element


def createLabeledFile(testProb, src, dst):
    writeCSVGZIP(labelTestAndTrain(parseCSVGZIP(src), testProb), dst)


def splitLabeledFile(src, testDst, trainDst):
    writeCSVGZIP(parseCSVGZIP(src), testDst, {"Usage": "Test"})
    writeCSVGZIP(parseCSVGZIP(src), trainDst, {"Usage": "Train"})


datasetPath = "rating-Vermont.csv.gz"
trainDatasetPath = "train_" + datasetPath
testDatasetPath = "test_" + datasetPath
labeledDatasetPath = "labeled_" + datasetPath

#createLabeledFile(0.33, datasetPath, "labeled_" + datasetPath)

#splitLabeledFile("labeled_" + datasetPath, "test_" + datasetPath, "train_" + datasetPath)

def loadDataset(path, len=None):
    reviews = list(parseCSVGZIP(datasetPath))[0:len]

    reviews = [{k: (v if k != 'rating' and k != 'timestamp' else (int(v) if k == 'rating' else int(v))) for k, v in d.items() if k != 'Usage'} for d in reviews]

    #reviews_tf = tf.data.Dataset.prefetch(tf.data.Dataset.from_tensor_slices(reviews), buffer_size=tf.data.AUTOTUNE)
    reviews_tf = tf.data.Dataset.from_tensor_slices(pd.DataFrame.from_dict(reviews).to_dict(orient="list")).map(lambda x: x)

    items_tf = reviews_tf.map(lambda x: x["business"])

    #items_tf = tf.data.Dataset.from_tensor_slices(pd.DataFrame.from_dict(items).to_dict(orient="list"))

    timestamps = np.concatenate(list(reviews_tf.map(lambda x: x["timestamp"]).batch(100)))

    max_timestamp = timestamps.max()
    min_timestamp = timestamps.min()

    timestamp_buckets = np.linspace(
        min_timestamp, max_timestamp, num=1000,
    )

    unique_items_ids = np.unique(np.concatenate(list(items_tf.batch(1_000))))
    unique_user_ids = np.unique(np.concatenate(list(reviews_tf.batch(1_000).map(
        lambda x: x["user"]))))

    return reviews_tf, items_tf, unique_items_ids, unique_user_ids, timestamp_buckets, timestamps


class UserModel(tf.keras.Model):

    def __init__(self, unique_user_ids, timestamp_buckets, timestamps, use_timestamps=True):
        super().__init__()

        self._use_timestamps = use_timestamps

        self.user_embedding = tf.keras.Sequential([
            tf.keras.layers.StringLookup(
                vocabulary=unique_user_ids, mask_token=None),
            tf.keras.layers.Embedding(len(unique_user_ids) + 1, 32),
        ])

        if use_timestamps:
            self.timestamp_embedding = tf.keras.Sequential([
                tf.keras.layers.Discretization(timestamp_buckets.tolist()),
                tf.keras.layers.Embedding(len(timestamp_buckets) + 1, 32),
            ])
            self.normalized_timestamp = tf.keras.layers.Normalization(
                axis=None
            )

            self.normalized_timestamp.adapt(timestamps)

    def call(self, inputs):
        if not self._use_timestamps:
            return self.user_embedding(inputs["user"])

        return tf.concat([
            self.user_embedding(inputs["user"]),
            self.timestamp_embedding(inputs["timestamp"]),
            tf.reshape(self.normalized_timestamp(inputs["timestamp"]), (-1, 1)),
        ], axis=1)


class MovieModel(tf.keras.Model):

    def __init__(self, unique_items_ids, items):
        super().__init__()

        max_tokens = 10_000

        self.title_embedding = tf.keras.Sequential([
            tf.keras.layers.StringLookup(
                vocabulary=unique_items_ids, mask_token=None),
            tf.keras.layers.Embedding(len(unique_items_ids) + 1, 32)
        ])

        self.title_vectorizer = tf.keras.layers.TextVectorization(
            max_tokens=max_tokens)

        self.title_text_embedding = tf.keras.Sequential([
            self.title_vectorizer,
            tf.keras.layers.Embedding(max_tokens, 32, mask_zero=True),
            tf.keras.layers.GlobalAveragePooling1D(),
        ])

        self.title_vectorizer.adapt(items)

    def call(self, titles):
        return tf.concat([
            self.title_embedding(titles),
            self.title_text_embedding(titles),
        ], axis=1)


class MovielensModel(tfrs.models.Model):

  def __init__(self, unique_items_ids, items, timestamp_buckets, timestamps, use_timestamps=True):
    super().__init__()
    self.query_model = tf.keras.Sequential([
      UserModel(unique_user_ids, timestamp_buckets, timestamps, use_timestamps),
      tf.keras.layers.Dense(32),
      tf.keras.layers.Dense(32),
      tf.keras.layers.Dense(32)
    ])
    self.candidate_model = tf.keras.Sequential([
      MovieModel(unique_items_ids, items),
      tf.keras.layers.Dense(32),
      tf.keras.layers.Dense(32),
      tf.keras.layers.Dense(32)
    ])
    self.task = tfrs.tasks.Retrieval(
        metrics=tfrs.metrics.FactorizedTopK(
            candidates=items.batch(128).map(self.candidate_model),
        ),
    )

  def compute_loss(self, features, training=False):
    # We only pass the user id and timestamp features into the query model. This
    # is to ensure that the training inputs would have the same keys as the
    # query inputs. Otherwise the discrepancy in input structure would cause an
    # error when loading the query model after saving it.
    query_embeddings = self.query_model({
        "user": features["user"],
        "timestamp": features["timestamp"],
    })
    movie_embeddings = self.candidate_model(features["business"])

    return self.task(query_embeddings, movie_embeddings)


#print(device_lib.list_local_devices())

#exit(0)

#print(tf.__version__)
#print(tf.config.list_physical_devices())
#print(tf.config.list_logical_devices())


reviews_tf, items_tf, unique_items_ids, unique_user_ids, timestamp_buckets, timestamps = loadDataset(trainDatasetPath, 1000)
test_reviews_tf, test_items_tf, test_unique_items_ids, test_unique_user_ids, test_timestamp_buckets, test_timestamps = loadDataset(testDatasetPath, 100)

np.concatenate((unique_user_ids, test_unique_user_ids))
unique_user_ids = np.unique(unique_user_ids)
np.concatenate((unique_items_ids, test_unique_items_ids))
unique_items_ids = np.unique(unique_items_ids)

print(unique_user_ids.shape)
print(unique_items_ids.shape)

model = MovielensModel(unique_items_ids, items_tf, timestamp_buckets, timestamps)
model.compile(optimizer=tf.keras.optimizers.Adagrad(0.1))

'''
model.compile(
    optimizer=tf.keras.optimizers.RMSprop(),  # Optimizer
    # Loss function to minimize
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    # List of metrics to monitor
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)
'''

cached_train = reviews_tf.batch(2048)
cached_test = test_reviews_tf.batch(2048)

#["factorized_top_k/top_100_categorical_accuracy"]

train_accuracy = model.evaluate(
    cached_train, return_dict=True)
test_accuracy = model.evaluate(
    cached_test, return_dict=True)

print(f"Top-100 accuracy (train): {train_accuracy}.")
print(f"Top-100 accuracy (test): {test_accuracy}.")

model.fit(cached_train, epochs=100, batch_size=1024, use_multiprocessing=True)

train_accuracy = model.evaluate(
    cached_train, return_dict=True)
test_accuracy = model.evaluate(
    cached_test, return_dict=True)

print(f"Top-100 accuracy (train): {train_accuracy}.")
print(f"Top-100 accuracy (test): {test_accuracy}.")

#model.predict()
