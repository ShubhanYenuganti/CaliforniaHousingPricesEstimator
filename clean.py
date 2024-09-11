import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


import tensorflow as tf


data = pd.read_csv('/Users/shubhan/Desktop/California_Real_Estate.csv')

# Remove Blank Data
data = data.dropna(axis = 0)

# Handle Cities
# Load Unique Cities into a pandas DataFrame
unique_cities = data['city'].unique()
cities = pd.DataFrame(unique_cities, columns = ['city'])

# Encode city names as integers
city_lookup = tf.keras.layers.StringLookup()
city_lookup.adapt(cities['city'])

data['city_encoded'] = city_lookup(data['city'])

data = data.drop(['city'], axis = 1)

# Encode prev_sold_date
data['prev_sold_date'] = pd.to_datetime(data['prev_sold_date'])

data['prev_sold_year'] = data['prev_sold_date'].dt.year
data['prev_sold_month'] = data['prev_sold_date'].dt.month

data = data.drop(['prev_sold_date'], axis = 1)

# Outliers
q = data['price'].quantile(0.98)
data = data[data['price'] < q]

q = data['acre_lot'].quantile(0.98)
data = data[data['acre_lot'] < q]

q = data['house_size'].quantile(0.98)
data = data[data['house_size'] < q]

q = data['prev_sold_year'].quantile(0.02)
data = data[data['prev_sold_year'] > q]

q = data['bed'].quantile(0.98)
data = data[data['bed'] < q]

q = data['bath'].quantile(0.98)
data = data[data['bath'] < q]

sns.distplot(data['price'])
plt.show()
sns.distplot(data['acre_lot'])
plt.show()
sns.distplot(data['house_size'])
plt.show()
sns.distplot(data['prev_sold_year'])
plt.show()
sns.distplot(data['bed'])
plt.show()
sns.distplot(data['bath'])
plt.show()

print(data['price'].describe())
print(data['acre_lot'].describe())
print(data['house_size'].describe())
print(data['prev_sold_year'].describe())
print(data['bed'].describe())
print(data['bath'].describe())

# Update CSV File with Cleaned Data
file_path = '/Users/shubhan/Desktop/California_Real_Estate_Updated.xlsx'

with pd.ExcelWriter(file_path, mode='a', engine='openpyxl') as writer:
    data.to_excel(writer, sheet_name='Sheet', index=False)