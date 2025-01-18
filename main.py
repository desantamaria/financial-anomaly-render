from fastapi import FastAPI
import pickle
import pandas as pd
import uvicorn
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf



# Initialize FastAPI
app = FastAPI()

# Function to load model
def load_model(filename):
    with open(filename, "rb") as file:
        return pickle.load(file)

# Load Models
xgboost_model = load_model('xgboost.pkl')
lr_model = load_model('lr_model.pkl')
forest = load_model('forest.pkl')

# Preprocess given data into a dataframe
def preprocess_data(input_dict):
    # Extract relevant features from the input dictionary
    features = ['VIX', 'DXY', 'JPY', 'GBP', 'MXEU', 'MXCN', 'VIX_MA', 'DXY_MA', 'JPY_MA', 'GBP_MA', 'MXEU_MA', 'MXCN_MA']
    data = {feature: [input_dict.get(feature, None)] for feature in features}

    df = pd.DataFrame(data)
    return df


# Get predictions and probabilities from all models and return average
def get_prediction(transaction_dict):
    preprocessed_data = preprocess_data(transaction_dict)
    print(preprocessed_data)

    # Get predictions from all models
    predictions = {
        'XGBoost': xgboost_model.predict(preprocessed_data)[0],
        'Logistic Regression': lr_model.predict(preprocessed_data)[0],
        'Isolation Forest': forest.predict(preprocessed_data)[0]
    }

    # Get probabilities from models that support predict_proba
    probabilities = {
        'XGBoost': xgboost_model.predict_proba(preprocessed_data)[0][1],
        # 'Logistic Regression': lr_model.predict_proba(preprocessed_data)[0][1],
        # 'Isolation Forest': forest.predict_proba(preprocessed_data)[0][1]
    }

    return predictions, probabilities


# Endpoint to get predictions and probabilities
@app.post("/predict")
async def predict(data: dict):
    prediction, probabilities = get_prediction(data)

    # Convert NumPy types to Python types
    prediction = {model: int(pred) for model, pred in prediction.items()}
    probabilities = {model: float(prob) for model, prob in probabilities.items()}
    
    print(prediction)

    return {
        "prediction": prediction,
        "probability": probabilities
    }


def createMA(df, window, ticker):
    df[f'{ticker}_MA'] = df[ticker].rolling(window=window).mean()
    return df

@app.get("/getMarketData")
async def get_market_data():
    try:
        ticker_list = ['VIX', 'DXY', 'JPY', 'GBP', 'MXEU', 'MXCN']
        yf_tickers = ['^VIX', 'DX-Y.NYB', 'JPY=X', 'GBPUSD=X', 'MXEU.L', 'MXCN.SW']
        
        df = yf.download(yf_tickers, period='6mo')
        column_mapping = dict(zip(yf_tickers, ticker_list))
        df = df.rename(columns=column_mapping)
        df = df[['Close']].droplevel(level=0, axis=1)
        df = df.dropna()
        
        for ticker in ticker_list:
            df = createMA(df, 12, ticker)
        
        df = df.dropna()
        df = df.reset_index()
        
        json_data = {
            ticker: [{'date': str(date), 'price': float(price), 'MA': float(ma)} 
                    for date, price, ma in zip(df['Date'], df[ticker], df[f'{ticker}_MA'])]
            for ticker in ticker_list
        }
        
        dates = {
            'dates': [str(date) for date in df['Date']]
        }
        
        return {**json_data, **dates}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
# Run the app
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)