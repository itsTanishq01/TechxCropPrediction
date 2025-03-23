import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os
import numpy as np

app = FastAPI()

# Load dataset
file_path = "filtered_agridata.csv"
df = pd.read_csv(file_path)
df['date'] = pd.to_datetime(df['date'])
df_cleaned = df.dropna().copy()

# Request models
class CommodityRequest(BaseModel):
    commodity_name: str

class PredictionRequest(BaseModel):
    commodity_name: str
    min_price: float
    max_price: float

class FuturePredictionRequest(BaseModel):
    commodity_name: str
    future_days: int

# Train model function
def train_model_for_crop(commodity_name):
    crop_data = df_cleaned[df_cleaned['commodity_name'] == commodity_name].copy()
    
    if crop_data.empty:
        return {"error": f"No data found for commodity: {commodity_name}. Please check the dataset."}

    crop_data.sort_values(by='date', inplace=True)
    
    X = crop_data[['min_price', 'max_price']]
    y = crop_data['modal_price']
    
    if X.shape[0] < 2:  # Ensure there are enough samples for splitting
        return {"error": "Not enough data to train the model. Please provide more data."}

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    model_filename = f"{commodity_name}_price_model.pkl"
    joblib.dump(model, model_filename)

    return {
        "commodity_name": commodity_name,
        "MSE": mse,
        "MAE": mae,
        "R²": r2,
        "model_file": model_filename
    }

# API Endpoints
@app.post("/train/")
def train_model(request: CommodityRequest):
    return train_model_for_crop(request.commodity_name)

@app.post("/predict/")
def predict_price(request: PredictionRequest):
    model_filename = f"{request.commodity_name}_price_model.pkl"
    if not os.path.exists(model_filename):
        return {"error": "Model not found. Please train the model first."}
    
    model = joblib.load(model_filename)
    
    input_data = [[request.min_price, request.max_price]]
    predicted_price = model.predict(input_data)[0]
    
    return {
        "commodity_name": request.commodity_name,
        "predicted_modal_price": predicted_price
    }

@app.post("/predict_future/")
def predict_future_prices(request: FuturePredictionRequest):
    model_filename = f"{request.commodity_name}_price_model.pkl"
    if not os.path.exists(model_filename):
        return {"error": "Model not found. Please train the model first."}
    
    model = joblib.load(model_filename)
    
    historical_data = df_cleaned[df_cleaned['commodity_name'] == request.commodity_name]
    if historical_data.empty:
        return {"error": "No data available for future prediction."}
    
    avg_min_price = historical_data['min_price'].mean()
    avg_max_price = historical_data['max_price'].mean()
    
    future_prices = []
    for i in range(request.future_days):
        min_price_future = avg_min_price * (1 + 0.01 * i)
        max_price_future = avg_max_price * (1 + 0.01 * i)
        predicted_price = model.predict([[min_price_future, max_price_future]])[0]
        future_prices.append({"day": i + 1, "predicted_price": predicted_price})
    
    return {
        "commodity_name": request.commodity_name,
        "future_prices": future_prices
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)
