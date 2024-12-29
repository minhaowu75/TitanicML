import numpy as np
import os
import pandas as pd

import tensorflow as tf
import ydf

train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")

def preprocess(df):
    df = df.copy()

    def normalize_name(x):
        return " ".join([v.strip(",()[].\"'") for v in x.split(" ")])
    
    def ticket_number(x):
        return x.split(" ")[-1]
    
    def ticket_item(x):
        items = x.split(" ")
        if len(items) == 1:
            return "NONE"
        return "_".join(items[0:-1])
    
    df["Name"] = df["Name"].apply(normalize_name)
    df["Ticket_number"] = df["Ticket"].apply(ticket_number)
    df["Ticket_item"] = df["Ticket"].apply(ticket_item)

    return df

def tokenize_names(features, labels=None):
    features["Name"] = tf.strings.split(features["Name"])
    return features, labels

preprocessed_train_df = preprocess(train_df)
preprocessed_test_df = preprocess(test_df)

input_features = list(preprocessed_train_df.columns)
input_features.remove("Ticket")
input_features.remove("PassengerId")
input_features.remove("Survived")
print(preprocessed_test_df)

ydf
model = ydf.GradientBoostedTreesLearner(label="Survived", num_trees=512).train(preprocessed_train_df)
model.evaluate(preprocessed_test_df)
# train_ds = ydf.keras.pd_dataframe_to_tf_dataset(preprocessed_train_df, label="Survived").map(tokenize_names)
# test_ds = ydf.keras.pd_dataframe_to_tf_dataset(preprocessed_test_df).map(tokenize_names)

# model = ydf.keras.GradientBoostedTreesModel(
#     verbose=0,
#     features=[ydf.keras.FeatureUsage(name=n) for n in input_features],
#     exclude_non_specified_features=True,
#     random_seed=1234,
# )

# model.fit(train_ds)

# self_evaluation = model.make_inspector().evaluation()
# print(f"Accuracy: {self_evaluation.accuracy} Loss:{self_evaluation.loss}")
