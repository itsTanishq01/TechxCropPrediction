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

class CommodityRequest(BaseModel):
    commodity_name: str

class PredictionRequest(BaseModel):
    commodity_name: str
    min_price: float
    max_price: float

class FuturePredictionRequest(BaseModel):
    commodity_name: str
    future_days: int

def train_model_for_crop(commodity_name):
    # Filter dataset for the selected commodity
    crop_data = df_cleaned[df_cleaned['commodity_name'] == commodity_name].copy()
    crop_data.sort_values(by='date', inplace=True)
    
    # Define features and target
    X = crop_data[['min_price', 'max_price']]
    y = crop_data['modal_price']
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Evaluate model
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Save model
    model_filename = f"{commodity_name}_price_model.pkl"
    joblib.dump(model, model_filename)
    
    return {
        "commodity_name": commodity_name,
        "MSE": mse,
        "MAE": mae,
        "R²": r2,
        "model_file": model_filename
    }

@app.post("/train/")
def train_model(request: CommodityRequest):
    return train_model_for_crop(request.commodity_name)

@app.post("/predict/")
def predict_price(request: PredictionRequest):
    model_filename = f"{request.commodity_name}_price_model.pkl"
    if not os.path.exists(model_filename):
        return {"error": "Model not found. Please train the model first."}
    
    # Load trained model
    model = joblib.load(model_filename)
    
    # Predict modal price
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
    
    # Load trained model
    model = joblib.load(model_filename)
    
    # Generate synthetic future min and max prices (simple trend-based approach)
    historical_data = df_cleaned[df_cleaned['commodity_name'] == request.commodity_name]
    avg_min_price = historical_data['min_price'].mean()
    avg_max_price = historical_data['max_price'].mean()
    
    future_prices = []
    for i in range(request.future_days):
        min_price_future = avg_min_price * (1 + 0.01 * i)  # Assuming a 1% daily increase
        max_price_future = avg_max_price * (1 + 0.01 * i)
        predicted_price = model.predict([[min_price_future, max_price_future]])[0]
        future_prices.append({"day": i + 1, "predicted_price": predicted_price})
    
    return {
        "commodity_name": request.commodity_name,
        "future_prices": future_prices
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)