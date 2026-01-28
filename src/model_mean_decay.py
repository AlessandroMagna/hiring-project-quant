import pandas as pd
import numpy as np
import json
import os
from requests_cache import CachedSession
from typer import run
from typing import Literal, Dict
from enum import StrEnum
from tqdm import tqdm

# DEFAULT PARAMETERS (Fallback)
# Safety net: if the optimizer hasn't run yet or the file is missing,
DEFAULT_CONFIG = {
    "N_MINUTES_INPUT": 60,
    "DEFAULT_LAMBDA": 0.98,
    "MIN_LAMBDA": 0.80,
    "MAX_LAMBDA": 0.995
}

class BiddingZone(StrEnum):
    DK1 = "DK1"
    DK2 = "DK2"

def load_config(bidding_zone: str) -> Dict:
    """
    Load optimal params from JSON, fallback to defaults.
    """
    path = f"config/best_params_{bidding_zone}.json"
    
    if os.path.exists(path):
        with open(path, "r") as f:
            optimized = json.load(f)
            # Merge defaults just in case the JSON is missing a key
            config = DEFAULT_CONFIG.copy()
            
            # Mapping lower_case optuna keys to UPPER_CASE internal config keys
            config["MIN_LAMBDA"] = optimized.get("min_lambda", config["MIN_LAMBDA"])
            config["MAX_LAMBDA"] = optimized.get("max_lambda", config["MAX_LAMBDA"])
            config["N_MINUTES_INPUT"] = optimized.get("n_minutes_input", config["N_MINUTES_INPUT"])
            
            print(f"Loaded optimized parameters for {bidding_zone}")
            return config
    else:
        print(f"Using default parameters for {bidding_zone}")
        return DEFAULT_CONFIG

def remove_outliers(series: pd.Series, threshold: float = 1000.0) -> pd.Series:
    """
    Rimuove gli outlier (> threshold in valore assoluto).
    Li sostituisce con l'interpolazione lineare.
    """
    clean_series = series.copy()
    
    # 1000MW is a massive spike for these zones, not interested to model these events
    is_outlier = clean_series.abs() > threshold
    
    if is_outlier.any():
        clean_series.loc[is_outlier] = np.nan
        # Linear interpolation fills the gap smoothly between the previous and next valid point
        clean_series = clean_series.interpolate(method='linear')
        # Fix edge cases (start/end of series)
        clean_series = clean_series.ffill().bfill()
        
    return clean_series

def get_recent_afrr_activation(
    bidding_zone: BiddingZone, 
    time_from: pd.Timestamp | None = None,
    limit: int = 100
) -> pd.Series:
    url = "https://api.energidataservice.dk/dataset/PowerSystemRightNow"
    params = {"limit": limit}
    
    if time_from is not None:
        params["start"] = time_from.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M")
        params["sort"] = "Minutes1UTC ASC"

    session = CachedSession(cache_name=".cache", expire_after=300) 
    response = session.get(url, params=params)
    response.raise_for_status()
    res_json = response.json()

    afrr_activation = (
        pd.Series(
            {
                pd.Timestamp(r["Minutes1UTC"]): r[f"aFRR_Activated{bidding_zone}"]
                for r in res_json["records"]
            }
        )
        .astype(float)
        .dropna()
        .sort_index(ascending=True) 
    )
    
    if not afrr_activation.empty:
        #standardize to CET since the API gives UTC
        afrr_activation.index = afrr_activation.index.tz_localize("UTC").tz_convert("CET")

    # Clean the data so model don't fit on noise
    afrr_activation = remove_outliers(afrr_activation)

    return afrr_activation

def get_forecast_horizon(now: pd.Timestamp, periods: int = 8) -> pd.DatetimeIndex:
    #predict  next 8 quarters (2 hours)
    return pd.date_range(start=now.ceil("15min"), periods=periods, freq="15min")

def calculate_dynamic_lambda(recent_series: pd.Series, config: Dict) -> float:
    # Need enough data  to trust the correlation
    if len(recent_series) < 10:
        return config["DEFAULT_LAMBDA"]
        
    # High value = strong trend (slow decay). Low value = noise (fast decay).
    autocorr = recent_series.autocorr(lag=1)
    
    if np.isnan(autocorr):
        return config["DEFAULT_LAMBDA"]
        
    return np.clip(autocorr, config["MIN_LAMBDA"], config["MAX_LAMBDA"])

def predict(
    forecast_horizon: pd.DatetimeIndex, 
    recent_activations: pd.Series,
    config: Dict
) -> pd.Series:
    
    if recent_activations.empty:
        return pd.Series(dtype=float)

    #starting point (t=0)
    current_val = recent_activations.iloc[-1]
    start_time = recent_activations.index[-1]
    
    # Ccalculate how fast we should revert to mean (0)
    decay_lambda = calculate_dynamic_lambda(recent_activations, config)
    
    # simulate minute-by-minute decay
    future_minutes = 120
    future_index = pd.date_range(start=start_time + pd.Timedelta(minutes=1), periods=future_minutes, freq="1min")
    
    #exponential decay: y_t = y_0 * lambda^t
    steps = np.arange(1, future_minutes + 1)
    decay_curve = current_val * (decay_lambda ** steps)
    
    #resampleinto 15-min settlement periods
    raw_prediction = pd.Series(decay_curve, index=future_index)
    resampled_pred = raw_prediction.resample("15min").mean()
    
    # slign to the forecast horizon 
    final_prediction = resampled_pred.reindex(forecast_horizon).ffill().bfill()
    
    return final_prediction

def backtest(activations: pd.Series, params: Dict = None):
    """
    Backtest accepting optional params override for optimization.
    """
    # If 'params' is provided (by Optuna), use those. Otherwise load defaults.
    config = params if params else DEFAULT_CONFIG
    
    print(f"Backtesting with: {config}")
    
    target = activations.resample("15min").mean().sort_index()
    activations = activations.sort_index(ascending=True)
    predictions = []
    
    input_window = config["N_MINUTES_INPUT"]
    
    iterator = activations.index if params else tqdm(activations.index)

    for prediction_time in iterator:
        history_slice = activations.loc[:prediction_time]
        
        # need  full window to calculate  autocorrelation
        if len(history_slice) < input_window:
            continue

        recent_activations = history_slice.iloc[-input_window:]
        forecast_horizon = get_forecast_horizon(prediction_time + pd.Timedelta("1min"))
        
        prediction = predict(forecast_horizon, recent_activations, config)
        
        if prediction.empty:
            continue

        prediction_df = prediction.to_frame("prediction").reset_index(names="delivery_start")
        prediction_df["prediction_time"] = prediction_time
        predictions.append(prediction_df)

    if not predictions:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    predictions = pd.concat(predictions)
    predictions["time_to_delivery"] = predictions.eval("delivery_start - prediction_time")

    data = predictions.merge(
        target.to_frame("target"),
        left_on="delivery_start",
        right_index=True,
        how="inner",
    )
    
    data["abs_error"] = data.eval("target - prediction").abs()
    data["sign_correct"] = np.sign(data.target) == np.sign(data.prediction)

    mae = data.groupby("time_to_delivery").abs_error.mean()
    sign_accuracy = data.groupby("time_to_delivery").sign_correct.mean()
    
    return mae, sign_accuracy

def main(
    task: Literal["predict", "backtest"],
    bidding_zone: BiddingZone,
    time_from: str | None = None,
):
    # pull  best params found by the optimizer
    config = load_config(bidding_zone)

    match task:
        case "predict":
            now = pd.Timestamp("now", tz="CET")
            forecast_horizon = get_forecast_horizon(now)
            
            print(f"--- Forecast for {bidding_zone} at {now} ---")

            input_window = config["N_MINUTES_INPUT"]
            recent_activations = get_recent_afrr_activation(
                bidding_zone=bidding_zone, 
                limit=input_window + 10
            )
            
            if len(recent_activations) < input_window:
                print("Error: Not enough data.")
                return
                
            recent_activations = recent_activations.iloc[-input_window:]
            prediction = predict(forecast_horizon, recent_activations, config)

            print(prediction.to_string())
            
        case "backtest":
            assert time_from is not None, "time_from required"
            time_from = pd.Timestamp(time_from, tz="CET")
            
            print(f"Fetching backtest data starting from {time_from}...")
            activations = get_recent_afrr_activation(
                bidding_zone=bidding_zone, 
                time_from=time_from,
                limit=5000 
            )
            
            # Pass the loaded config to the backtest
            mae, sign_accuracy = backtest(activations, params=config)

            print(f"Backtest results for {bidding_zone}")
            print("Mean absolute error:")
            print(mae.to_string())
            print("Sign accuracy:")
            print(sign_accuracy.to_string())

if __name__ == "__main__":
    run(main)