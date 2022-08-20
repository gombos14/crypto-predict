import numpy as np
import pandas_datareader as web
import datetime as dt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential


def get_data(crypto, currency, start_date: dt.datetime, end_date: dt.datetime):
    data = web.DataReader(f"{crypto}-{currency}", "yahoo", start_date, end_date)
    return data


def get_scaler(min, max):
    return MinMaxScaler(feature_range=(min, max))


def prepare_data(data, prediction_days):
    x_train, y_train = [], []

    scaler = get_scaler(0, 1)
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

    for x in range(prediction_days, len(scaled_data)):
        x_train.append(scaled_data[x - prediction_days : x, 0])
        y_train.append(scaled_data[x, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    return x_train, y_train


def create_neural_network(x_train, y_train):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, epochs=25, batch_size=32)

    return model
