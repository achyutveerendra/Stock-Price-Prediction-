from flask import Flask, request, jsonify, render_template
import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import pickle
import datetime

app = Flask(__name__, static_url_path='', static_folder='static', template_folder='templates')


# Load the model and scaler
print("Loading model...")
model = load_model('model/stock_model.h5')
print("Model loaded.")
print("Loading scaler...")
with open('model/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
print("Scaler loaded.")

# Fetch historical data for a specific date
def fetch_historical_data(ticker, target_date):
    intervals = ['1m', '5m', '1d']
    for interval in intervals:
        try:
            print(f"Fetching {interval} data for ticker: {ticker} on date: {target_date}")
            start = datetime.datetime.strptime(target_date, '%Y-%m-%d')
            end = start + datetime.timedelta(days=1)
            
            data = yf.download(ticker, start=start, end=end, interval=interval)
            if not data.empty:
                print(f"{interval} data fetched successfully.")
                return data
        except Exception as e:
            print(f"Exception occurred while fetching {interval} data: {e}")
    
    print(f"No data found for ticker {ticker} on date {target_date}.")
    return None

# Predict using historical data
def predict_historical(ticker, target_date):
    print(f"Predicting for ticker: {ticker} on date: {target_date}")
    historical_data = fetch_historical_data(ticker, target_date)
    if historical_data is None:
        print(f"Historical data for {ticker} on {target_date} is None.")
        return None

    data = historical_data[['Close']].values
    if len(data) == 0:
        print("No closing price data available.")
        return None

    scaled_data = scaler.transform(data)
    seq_length = 60
    if len(scaled_data) < seq_length:
        print("Not enough data to make a prediction.")
        return None  # Not enough data to make a prediction

    X_test = []
    for i in range(seq_length, len(scaled_data)):
        X_test.append(scaled_data[i-seq_length:i, 0])
    X_test = np.array(X_test)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    predictions = model.predict(X_test)
    timestamps = historical_data.index[seq_length:].strftime('%Y-%m-%d %H:%M:%S').tolist()
    print("Prediction successful.")
    return [(timestamp, float(price)) for timestamp, price in zip(timestamps, scaler.inverse_transform(predictions).flatten())]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET'])
def predict():
    ticker = request.args.get('ticker', default='AAPL', type=str)
    target_date = request.args.get('date', default='2024-05-20', type=str)
    print(f"Received prediction request for ticker: {ticker} on date: {target_date}")
    if not ticker or not target_date:
        print("Missing ticker or date parameter.")
        return jsonify({'error': 'Missing ticker or date parameter.'}), 400
    try:
        datetime.datetime.strptime(target_date, '%Y-%m-%d')
    except ValueError:
        print("Invalid date format.")
        return jsonify({'error': 'Invalid date format. Use YYYY-MM-DD.'}), 400

    predictions = predict_historical(ticker, target_date)
    if predictions is None:
        print("No predictions available.")
        return jsonify({'error': 'No data available for the given ticker or not enough data to make a prediction.'}), 400

    print("Returning predictions.")
    return jsonify(predictions)

if __name__ == '__main__':
    app.run(debug=True)
