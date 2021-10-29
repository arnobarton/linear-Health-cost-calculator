!pip install -q git+https://github.com/tensorflow/docs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
  # %tensorflow_version only exists in Colab.
  %tensorflow_version 2.x
except Exception:
  pass
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling

from tensorflow.keras.layers.experimental import preprocessing
from sklearn.model_selection import train_test_split

import seaborn as sns



# Import data
!wget https://cdn.freecodecamp.org/project-data/health-costs/insurance.csv
dataset = pd.read_csv('insurance.csv')
dataset.tail()


dataset['sex'] = dataset['sex'].map({'male': 0, 'female': 1})
dataset['smoker'] = dataset['smoker'].map({'no': 0, 'yes': 1})
dataset['region'] = dataset['region'].map({'southeast': 0, 'northeast': 1, 'northwest': 2, 'southwest': 3})

sex_dummy = pd.get_dummies(dataset['sex'])
smoker_dummy = pd.get_dummies(dataset['smoker'])
region_dummy = pd.get_dummies(dataset['region'])
dataset = pd.concat([dataset,sex_dummy,smoker_dummy,region_dummy], axis=1)
  
dataset = dataset.drop(['sex','smoker','region'], axis=1)

# dataset.tail()

train_dataset = dataset.sample(frac=0.8, random_state=20)
test_dataset = dataset.drop(train_dataset.index)

train_features = train_dataset.copy()
test_features = test_dataset.copy()

train_labels = train_features.pop('expenses')
test_labels = test_features.pop('expenses')

normalizer = preprocessing.Normalization(input_shape=[11,], axis=None)
normalizer.adapt(np.array(train_features,train_labels))


def build_and_compile_model(norm):
  modelbruh = keras.Sequential([
      norm,
      layers.Dense(64, activation='relu'),
      layers.Dense(64, activation='relu'),
      layers.Dense(1)
  ])

  modelbruh.compile(loss='mean_absolute_error',
                optimizer=tf.keras.optimizers.Adam(0.1),
                metrics= ['mae','mse'])
  return modelbruh
model = build_and_compile_model(normalizer)


history = model.fit(
    train_features, train_labels,
    verbose=2, epochs=200)
test_results = []
loss = model.evaluate(test_features, test_labels, verbose=2)

# RUN THIS CELL TO TEST YOUR MODEL. DO NOT MODIFY CONTENTS.
# Test model by checking how well the model generalizes using the test set.
loss, mae, mse = model.evaluate(test_features, test_labels, verbose=2)

print("Testing set Mean Abs Error: {:5.2f} expenses".format(mae))

if mae < 3500:
  print("a")
else:
  print("The Mean Abs Error must be less than 3500. Keep trying.")

# Plot predictions.
test_predictions = model.predict(test_features).flatten()


a = plt.axes(aspect='equal')
plt.scatter(test_labels, test_predictions)
plt.xlabel('True values (expenses)')
plt.ylabel('Predictions (expenses)')
lims = [0, 50000]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims,lims)
