# Cryptocurrency Price Prediction Using LSTM

This project demonstrates how to build a basic Long Short-Term Memory (LSTM) neural network using TensorFlow/Keras to predict cryptocurrency prices, specifically Ethereum (ETH) against the US Dollar (USD).

## Features

* Fetch historical cryptocurrency data using Yahoo Finance.
* Preprocess the data using MinMaxScaler.
* Build and train an LSTM-based neural network.
* Visualize actual vs predicted price trends.
* Predict the next day's price.

## Project Structure

```
.
├── main.py         # Main script to train the model and visualize predictions
├── crypto.py       # Helper functions for data loading, preprocessing, and model creation
```

## Requirements

* Python 3.7+
* TensorFlow
* NumPy
* Pandas
* Matplotlib
* pandas\_datareader

You can install the dependencies using:

```bash
pip install -r requirements.txt
```

## Usage

1. **Clone the repository or copy the files locally.**
2. **Run the `main.py` script:**

   ```bash
   python main.py
   ```
3. **View the plot:** The script will generate a plot showing the actual vs predicted prices of ETH/USD.

## Customization

* Change the `CRYPTO` or `AGAINST` variables in `main.py` to try different coins or fiat currencies (e.g., BTC, LTC, EUR).
* Adjust the `PREDICTION_DAYS` to control how many past days are used for prediction.
* Modify the model architecture or training parameters in `crypto.py` for experimentation.

## Limitations

* The model is trained on past closing prices only and does not consider external market factors.
* Predictions are for educational purposes and are not suitable for financial decisions.

## License

This project is licensed under the MIT License.

---

*Created for educational purposes to demonstrate LSTM modeling for time series forecasting.*
