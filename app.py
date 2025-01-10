from flask import Flask, render_template, request
import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
import numpy as np

app = Flask(__name__)

# Fungsi untuk mengambil data dari Yahoo Finance
def get_crypto_data(symbol, start_date, end_date):
    data = yf.download(symbol, start=start_date, end=end_date)
    return data[['Close']]

# Fungsi untuk melatih model prediksi
def train_model(data):
    # Reset index untuk menghindari multi-index
    data = data.reset_index(drop=True)

    # Tambahkan kolom prediksi
    data['Prediction'] = data['Close'].shift(-1)
    data = data[:-1]  # Menghapus baris terakhir yang tidak memiliki label

    # Validasi kolom dan NaN
    if data['Close'].isnull().values.any():
        data = data.dropna(subset=['Close'])

    # Pastikan data tidak kosong
    if data.empty:
        raise ValueError("Data kosong setelah membersihkan NaN.")

    # Split data menjadi fitur dan label
    X = data[['Close']]
    y = data['Prediction']

    # Normalisasi data
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_scaled = scaler_X.fit_transform(X)  # Skalakan fitur (X)
    y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))  # Skalakan label (y)

    # Membagi data menjadi train-test
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, shuffle=False)

    # Model Random Forest Regressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train.ravel())

    # Prediksi pada X_test
    y_pred_scaled = model.predict(X_test)
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))  # Balikkan ke skala asli

    # Prediksi harga hari berikutnya
    last_close_price = np.array([[data['Close'].iloc[-1]]])  # Dimensi (1, 1)
    print("Last Close Price:", last_close_price)

    last_close_price_scaled = scaler_X.transform(last_close_price.reshape(-1, 1))  # Pastikan dimensi (1, 1)
    print("Last Close Price Scaled:", last_close_price_scaled)

    next_day_prediction_scaled = model.predict(last_close_price_scaled.reshape(1, -1))
    next_day_prediction = scaler_y.inverse_transform(next_day_prediction_scaled.reshape(-1, 1))[0][0]

    return model, X_test, y_test, y_pred, next_day_prediction

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        symbol = request.form['symbol']
        start_date = request.form['start_date']
        end_date = request.form['end_date']

        # Mendapatkan data
        crypto_data = get_crypto_data(symbol, start_date, end_date)
        print("Data yang diambil dari Yahoo Finance:")
        print(crypto_data.head())
        print("Apakah data kosong?:", crypto_data.empty)
        print("Apakah ada NaN di kolom 'Close'?:", crypto_data['Close'].isnull().any())

        if crypto_data.empty:
            return "Data kosong. Periksa simbol cryptocurrency atau rentang tanggal."

        # Melatih model
        model, X_test, y_test, y_pred, next_day_prediction = train_model(crypto_data)
        print("Prediksi untuk hari berikutnya:", next_day_prediction)

        # Membuat grafik menggunakan Plotly
        fig = go.Figure()

        # Reset indeks untuk grafik
        crypto_data = crypto_data.reset_index()

        # Harga aktual
        real_prices = crypto_data['Close'].values.flatten()  # Konversi ke 1D array
        fig.add_trace(go.Scatter(
            x=crypto_data['Date'],  # Gunakan kolom Date untuk grafik
            y=real_prices,
            mode='lines',
            name='Harga Real'
        ))

        # Harga prediksi
        predicted_prices = y_pred.flatten()  # Konversi ke 1D array
        fig.add_trace(go.Scatter(
            x=crypto_data['Date'][-len(predicted_prices):],  # Gunakan tanggal dari bagian akhir
            y=predicted_prices,
            mode='lines',
            name='Prediksi Harga'
        ))

        # Tambahkan layout untuk grafik
        fig.update_layout(
            title=f'Grafik Prediksi Harga untuk {symbol.upper()}',
            xaxis_title='Tanggal',
            yaxis_title='Harga (USD)',
            template='plotly_white'
        )

        # Simpan grafik sebagai HTML
        fig.write_html("static/plot.html")

        return render_template('index.html', plot_url="plot.html", next_day_prediction=next_day_prediction)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
