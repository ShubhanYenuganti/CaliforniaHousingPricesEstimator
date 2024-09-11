import numpy as np
from sklearn import preprocessing

raw_csv_data = np.loadtxt('/Users/shubhan/Desktop/California_Real_Estate_Updated.csv', delimiter = ',')

unscaled_inputs_all = raw_csv_data[:,:-1]
targets = raw_csv_data[:,-1]

# Standardize the Inputs
scaled_inputs = preprocessing.scale(unscaled_inputs_all)

# Shuffle the Data
shuffled_indices = np.arange(scaled_inputs.shape[0])
np.random.shuffle(shuffled_indices)

shuffled_inputs = scaled_inputs[shuffled_indices]
shuffled_targets = targets[shuffled_indices]

# Split the Dataset
total = shuffled_inputs.shape[0]
train_count = int(0.8*total)
validation_count = int(0.1*total)
test_count = total - train_count - validation_count

train_inputs = shuffled_inputs[:train_count]
train_targets = shuffled_targets[:train_count]

validation_inputs = shuffled_inputs[train_count:train_count + validation_count]
validation_targets = shuffled_targets[train_count:train_count + validation_count]

test_inputs = shuffled_inputs[train_count + validation_count:]
test_targets = shuffled_targets[train_count + validation_count:]

# Save datasets
np.savez('California_RealEstate_train', inputs = train_inputs, target = train_targets)
np.savez('California_RealEstate_validation', inputs = validation_inputs, target = validation_targets)
np.savez('California_RealEstate_test', inputs = test_inputs, target = test_targets)