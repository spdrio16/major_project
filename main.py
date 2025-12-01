from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import LSTM, Dense, Bidirectional
from keras.optimizers import Adam
import skfuzzy as fuzz
import matplotlib.pyplot as plt

import os

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# ------------- Fuzzy weight functions -------------
def gaussian_fuzzy(x, mean=0.5, sigma=0.15):
    return fuzz.gaussmf(np.array([x]), mean, sigma)[0]

def triangular_fuzzy(x, a=0.2, b=0.5, c=0.8):
    return fuzz.trimf(np.array([x]), [a, b, c])[0]

def sigmoid_fuzzy(x, a=10, c=0.5):
    return fuzz.sigmf(np.array([x]), c, a)[0]

def trapezoidal_fuzzy(x, a=0.2, b=0.4, c=0.6, d=0.8):
    return fuzz.trapmf(np.array([x]), [a, b, c, d])[0]

def apply_fuzzy_weights(data, method):
    weights = []
    for val in data:
        x = val[0]
        if method == 'gaussian': w = gaussian_fuzzy(x)
        elif method == 'triangular': w = triangular_fuzzy(x)
        elif method == 'sigmoid': w = sigmoid_fuzzy(x)
        elif method == 'trapezoidal': w = trapezoidal_fuzzy(x)
        weights.append([w])
    return np.array(weights)

def create_sequences(data, window=3):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i:i + window])
        y.append(data[i + window])
    return np.array(X), np.array(y)

def evaluate_model(X, y, model_type='lstm'):
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    if model_type == 'linear':
        X_lin = X.reshape((X.shape[0], X.shape[1]))
        model = LinearRegression().fit(X_lin[:split], y_train[:split])
        preds = model.predict(X_lin[split:])
    else:
        model = Sequential()
        if model_type == 'bilstm':
            model.add(Bidirectional(LSTM(50), input_shape=(X.shape[1], X.shape[2])))
        else:
            model.add(LSTM(50, input_shape=(X.shape[1], X.shape[2])))
        model.add(Dense(1))
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        model.fit(X_train, y_train, epochs=20, batch_size=16, verbose=0)
        preds = model.predict(X_test)

    mse = mean_squared_error(y_test, preds)
    return mse

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, model_type: str = Form(...)):
    # 1. Fetch data
    data = yf.download('GC=F', start='2020-01-01', end='2023-12-31')
    df = data[['Close']].rename(columns={'Close': 'Price'})
    df.dropna(inplace=True)

    if df.empty:
        return templates.TemplateResponse("index.html", {
            "request": request, 
            "error": "No data fetched. Check internet connection or symbol."
        })

    # 2. Normalize
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(df)

    # 3. Run fuzzy + models
    fuzzy_methods = ['gaussian', 'triangular', 'sigmoid', 'trapezoidal']
    results = {}
    for fuzzy in fuzzy_methods:
        fuzzy_weights = apply_fuzzy_weights(data_scaled, fuzzy)
        weighted_data = data_scaled * fuzzy_weights
        X, y = create_sequences(weighted_data)
        results[fuzzy] = {
            'lstm': evaluate_model(X, y, 'lstm'),
            'bilstm': evaluate_model(X, y, 'bilstm'),
            'linear': evaluate_model(X, y, 'linear')
        }

    # 4. Find best fuzzy for selected model
    best_fuzzy = min(results, key=lambda f: results[f][model_type])

    # 5. Calculate averages
    # Correct single value extraction
    today_price = round(df['Price'].iloc[-1].item(), 2)
    avg_3day = round(df['Price'].iloc[-3:].mean().item(), 2)
    avg_5day = round(df['Price'].iloc[-5:].mean().item(), 2)


    # 6. Plot graph
    plt.figure(figsize=(10,6))
    df['Price'].plot(label='Price', color='blue')
    df['Price'].rolling(3).mean().plot(label='3-day Avg', color='orange')
    df['Price'].rolling(5).mean().plot(label='5-day Avg', color='green')
    plt.title('Gold Price with Moving Averages')
    plt.legend()
    plt.grid()
    plot_path = 'static/gold_price_plot.png'
    plt.savefig(plot_path)
    plt.close()

    return templates.TemplateResponse("index.html", {
        "request": request,
        "selected_model": model_type,
        "results": results,
        "best_model": best_fuzzy,
        "today_price": today_price,
        "avg_3day": avg_3day,
        "avg_5day": avg_5day,
        "plot_path": plot_path
    })
