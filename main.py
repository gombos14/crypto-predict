import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt

from crypto import get_data, prepare_data, create_neural_network, get_scaler

CRYPTO = "ETH"
AGAINST = "USD"
PREDICTION_DAYS = 60

start = dt.datetime(2020, 1, 1)
end = dt.datetime.now()

data = get_data(CRYPTO, AGAINST, start, end)
x_train, y_train = prepare_data(data, PREDICTION_DAYS)
model = create_neural_network(x_train, y_train)

# Test
test_start = dt.datetime(2020, 1, 1)
test_end = dt.datetime.now()

test_data = get_data(CRYPTO, AGAINST, test_start, test_end)
actual_prices = test_data['Close'].values

total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)

scaler = get_scaler(0, 1)

model_inputs = total_dataset[len(total_dataset) - len(test_data) - PREDICTION_DAYS:].values
model_inputs = model_inputs.reshape(-1, 1)
model_inputs = scaler.fit_transform(model_inputs)

x_test = []

for x in range(PREDICTION_DAYS, len(model_inputs)):
    x_test.append(model_inputs[x - PREDICTION_DAYS : x, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

prediction_prices = model.predict(x_test)
prediction_prices = scaler.inverse_transform(prediction_prices)

plt.plot(actual_prices, color='black', label='Actual prices')
plt.plot(prediction_prices, color='green', label='Predicted prices')
plt.title(f"{CRYPTO} price prediction")
plt.xlabel("Time")
plt.ylabel(f"Price ({AGAINST})")
plt.legend(loc='upper left')
plt.show()

# Predict next day

# real_data = [model_inputs[len(model_inputs) + 1 - prediction_days : len(model_inputs) + 1, 0]]
# real_data = np.array(real_data)
# real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

# prediction = model.predict(real_data)
# prediction = scaler.inverse_transform(prediction)
# print(prediction)
