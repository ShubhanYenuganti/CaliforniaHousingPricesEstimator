import numpy as np
import tensorflow as tf

from sklearn.metrics import mean_squared_error, r2_score

# Load Data

npz = np.load('California_RealEstate_train.npz')

train_inputs = npz['inputs'].astype(float)
train_targets = npz['target'].astype(float)

npz = np.load('California_RealEstate_validation.npz')

validation_inputs = npz['inputs'].astype(float)
validation_targets = npz['target'].astype(float)

npz = np.load('California_RealEstate_test.npz')

test_inputs = npz['inputs'].astype(float)
test_targets = npz['target'].astype(float)

# Create Model
input_size = 7
output_size = 1
hidden_layer_size = 100


model = tf.keras.Sequential([
    tf.keras.layers.Dense(hidden_layer_size, activation = 'relu'),
    tf.keras.layers.Dense(hidden_layer_size, activation = 'relu'),
    tf.keras.layers.Dense(hidden_layer_size, activation = 'relu'),
    tf.keras.layers.Dense(hidden_layer_size, activation = 'relu'),
    tf.keras.layers.Dense(hidden_layer_size, activation = 'relu'),
    tf.keras.layers.Dense(output_size)
])
model.compile(optimizer='adam', loss = 'huber', metrics = ['mae'])

# Set batch size
batch_size = 100

# Set Epochs
max_epochs = 200

early_stopping = tf.keras.callbacks.EarlyStopping(patience = 10)

model.fit(
    train_inputs,
    train_targets,
    batch_size = batch_size,
    epochs = max_epochs,
    callbacks = [early_stopping],
    validation_data = (validation_inputs, validation_targets),
    verbose = 2
)

# Evaluate the model
test_loss, test_mae = model.evaluate(test_inputs, test_targets, verbose=1)
print(f'Test loss (MSE): {test_loss:.2f}')
print(f'Test MAE: {test_mae:.2f}')

# Alternatively, calculate RMSE and R-squared after predictions
test_predictions = model.predict(test_inputs)
rmse = np.sqrt(mean_squared_error(test_targets, test_predictions))
r2 = r2_score(test_targets, test_predictions)

print(f'RMSE: {rmse:.2f}')
print(f'R-squared: {r2:.2f}')
