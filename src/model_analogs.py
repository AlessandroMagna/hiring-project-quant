import pandas as pd
import numpy as np
from requests_cache import CachedSession
from typer import run
from typing import Literal, List
from enum import StrEnum
from tqdm import tqdm

# --- CONFIG PARAMS ---
PAST_DAYS = 3            # Lookback window. 3 days
DELTA_HOURS = 1          # Search radius around the current hour (e.g. if now is 10am, search 9am-11pm)
N_MINUTES_INPUT = 60     # Input pattern length. Must be divisible by 15 for the aggregation logic
N_MINUTES_OUTPUT = 120   # How far out we predict
K_BEST = 10              # Ensemble size. Averaging top 10 smooths out noise

class BiddingZone(StrEnum):
    DK1 = "DK1"
    DK2 = "DK2"

# 0) get_euclidean_distance
def get_euclidean_distance(s1: pd.Series, s2: pd.Series) -> float:
    # Standard L2 norm. Fast enough for 1D arrays
    return np.linalg.norm(s1.values - s2.values)

#1) interpolate values where |activation| is > 1000 
def remove_outliers(series: pd.Series, threshold: float = 1000.0) -> pd.Series:
    """
    Remove outliers (> threshold in absolute value).
    Substitute with linear interpolation
    """
    # Safety copy to avoid side effects
    clean_series = series.copy()
    
    # Simple threshold filter. 1000MW is huge for DK1/DK2, definitely an error/spike
    is_outlier = clean_series.abs() > threshold
    
    if is_outlier.any():
        clean_series.loc[is_outlier] = np.nan
        # Linear interpolation is better than mean filling for time series
        clean_series = clean_series.interpolate(method='linear')
        # Handle edge cases (first/last elements)
        clean_series = clean_series.ffill().bfill()
        
    return clean_series

# 1) get_recent_afrr_activation
def get_recent_afrr_activation(
    bidding_zone: BiddingZone, 
    time_from: pd.Timestamp | None = None,
    limit: int = 100
) -> pd.Series:
    """
    Gets data for aFRR activations from Energinet.
    """
    url = "https://api.energidataservice.dk/dataset/PowerSystemRightNow"
    params = {"limit": limit}
    
    if time_from is not None:
        params["start"] = time_from.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M")
        params["sort"] = "Minutes1UTC ASC"

    # Cache 1h because backtests often hit this endpoint
    session = CachedSession(cache_name=".cache", expire_after=3600) 
    
    response = session.get(url, params=params)
    response.raise_for_status()
    res_json = response.json()

    series = pd.Series(
        {
            pd.Timestamp(r["Minutes1UTC"]): r[f"aFRR_Activated{bidding_zone}"]
            for r in res_json["records"]
        }
    ).astype(float).dropna().sort_index(ascending=True)
    
    if not series.empty:
        series.index = series.index.tz_localize("UTC").tz_convert("CET")
    
    # Clean up spikes
    series = remove_outliers(series)
    
    return series

# 2) get future afrr activations from given timestamps
def get_future_afrr_activation(
    idx: List[pd.Timestamp], 
    n_minutes: int, 
    bidding_zone: BiddingZone
) -> pd.DataFrame:
    """
    Downloads future aFRR activations.
    Includes robustness fixes for Timezone alignment.
    """
    now = pd.Timestamp("now", tz="CET")
    futures_dict = {}

    for start_time in idx:
        end_time = start_time + pd.Timedelta(minutes=n_minutes)
        
        if end_time > now:
            continue
            
        # Fetch with a small buffer to handle boundary issues
        series = get_recent_afrr_activation(
            bidding_zone=bidding_zone, 
            time_from=start_time, 
            limit=n_minutes + 20 
        )
        
        if series.empty:
            continue

        expected_index = pd.date_range(
            start=start_time + pd.Timedelta(minutes=1), 
            periods=n_minutes, 
            freq="1min"
        )

        #Time consistency check: reindexing fails if TZ objects differ (e.g. UTC vs CET)
        if series.index.tz is not None:
            expected_index = expected_index.tz_convert(series.index.tz)

        # Force alignment
        series_aligned = series.reindex(expected_index)
        series_filled = series_aligned.ffill().bfill()
        
        # If the gap was too big, skip this candidate
        if series_filled.isna().any():
            continue
            
        futures_dict[start_time] = series_filled.values

    return pd.DataFrame(futures_dict)

# 3) get candidares to for the analogs
def get_analog_candidates(
    past_days: int, 
    delta_hours: int, 
    n_minutes: int, 
    now: pd.Timestamp,
    bidding_zone: BiddingZone
) -> pd.DataFrame:
    """
    Finds historical candidates aggregated to 15-minute blocks.
    """
    candidates_dict = {}
    
    # Fetch one big chunk of history instead of looping API calls
    total_minutes_needed = past_days * 24 * 60 + 3000
    start_history_fetch = now - pd.Timedelta(days=past_days + 2)
    
    full_history = get_recent_afrr_activation(
        bidding_zone=bidding_zone, 
        time_from=start_history_fetch,
        limit=total_minutes_needed
    )

    if full_history.empty:
        return pd.DataFrame()

    # Resample to 15mins to reduce search space by 15x, to make euclidean distance calc faster
    history_15min = full_history.resample('15min', closed='right', label='right').mean()
    steps_input = int(n_minutes / 15)

    for day_ago in range(1, past_days + 1):
        center_time = now - pd.Timedelta(days=day_ago)
        center_time = center_time.round("15min") 

        search_start = center_time - pd.Timedelta(hours=delta_hours)
        search_end = center_time + pd.Timedelta(hours=delta_hours)
        
        valid_anchors = history_15min.loc[search_start : search_end]
        
        for t in valid_anchors.index:
            try:
                # Use integer indexing to make it faster
                loc_idx = history_15min.index.get_loc(t)
                start_idx = loc_idx - steps_input + 1
                
                if start_idx >= 0:
                    candidate_series = history_15min.iloc[start_idx : loc_idx + 1]
                    if len(candidate_series) == steps_input:
                        candidates_dict[t] = candidate_series.values
            except KeyError:
                continue
                        
    return pd.DataFrame(candidates_dict)

# 4) get_best_analogs
def get_best_analogs(
    k: int, 
    df_analog_candidates: pd.DataFrame, 
    recent_afrr_activations: pd.Series
) -> List[pd.Timestamp]:
    distances = {}
    target_values = recent_afrr_activations.values
    
    for col_name in df_analog_candidates.columns:
        candidate_values = df_analog_candidates[col_name]
        
        if len(candidate_values) != len(target_values):
            continue

        # Compare the shape of the current pattern vs candidate
        dist = get_euclidean_distance(
            pd.Series(target_values), 
            pd.Series(candidate_values.values)
        )
        distances[col_name] = dist
        
    # Sort and pick top K closest matches
    sorted_candidates = sorted(distances.items(), key=lambda item: item[1])
    return [item[0] for item in sorted_candidates[:k]]

# 5) get_forecast_horizon
def get_forecast_horizon(now: pd.Timestamp, periods: int = 8) -> pd.DatetimeIndex:
    return pd.date_range(start=now.ceil("15min"), periods=periods, freq="15min")

# 6) predict_analogs
def predict_analogs(
    forecast_horizon: pd.DatetimeIndex, 
    analogs_future: pd.DataFrame
) -> pd.Series:
    """
    Creates final prediction. 
    """
    if analogs_future.empty:
        return pd.Series(dtype=float)

    # Ensemble Averaging: Combines all K scenarios into one forecast
    ensemble_mean = analogs_future.mean(axis=1)
    
    # Map raw minutes to timestamps
    start_time = forecast_horizon[0]
    timestamps = pd.date_range(
        start=start_time, 
        periods=len(ensemble_mean), 
        freq="1min"
    )
    time_indexed_prediction = pd.Series(ensemble_mean.values, index=timestamps)
    
    # Aggregate back to 15m to match settlement periods
    resampled = time_indexed_prediction.resample("15min").mean()
    # Ensure perfect alignment with the requested horizon
    final_prediction = resampled.reindex(forecast_horizon).ffill().bfill()
    
    return final_prediction

#BACKTESTING

def backtest(activations: pd.Series, bidding_zone: BiddingZone):
    print(f"Starting optimized backtest on {len(activations)} data points (minute-by-minute)...")
    
    target = activations.resample("15min").mean().sort_index()
    activations = activations.sort_index(ascending=True)
    predictions = []
    
    # Run prediction for EVERY minute to generate smooth error curves
    test_indices = activations.index
    
    for prediction_time in tqdm(test_indices):
        
        #We predict based on the last CLOSED 15-min block
        # e.g. at 10:07, we match patterns ending at 10:00.
        effective_time = prediction_time.floor("15min")
        
        # 1. get history
        history_slice = activations.loc[:effective_time]
        if len(history_slice) < N_MINUTES_INPUT:
            continue

        # 2. extract and aggregate pattern
        raw_pattern = history_slice.iloc[-N_MINUTES_INPUT:]
        
        #resample to match the candidate database format (15min)
        recent_pattern_15min = raw_pattern.resample('15min', closed='right', label='right').mean()
        recent_pattern_15min = recent_pattern_15min.ffill().bfill()

        if len(recent_pattern_15min) != int(N_MINUTES_INPUT/15):
            continue

        # 3. Get Candidates
        df_candidates = get_analog_candidates(
            past_days=PAST_DAYS,
            delta_hours=DELTA_HOURS,
            n_minutes=N_MINUTES_INPUT,
            now=effective_time,
            bidding_zone=bidding_zone
        )
        
        if df_candidates.empty:
            continue

        # 4. Get Best Analogs
        best_analog_indices = get_best_analogs(
            k=K_BEST,
            df_analog_candidates=df_candidates,
            recent_afrr_activations=recent_pattern_15min
        )
        
        if not best_analog_indices:
            continue

        # 5. Get Future (1-min resolution)
        analogs_future = get_future_afrr_activation(
            idx=best_analog_indices,
            n_minutes=N_MINUTES_OUTPUT,
            bidding_zone=bidding_zone
        )
        
        if analogs_future.empty:
            continue
            
        # 6. Predict
        forecast_horizon = get_forecast_horizon(prediction_time + pd.Timedelta("1min"))
        prediction = predict_analogs(forecast_horizon, analogs_future)
        
        if prediction.empty:
            continue

        prediction_df = prediction.to_frame("prediction").reset_index(names="delivery_start")
        prediction_df["prediction_time"] = prediction_time
        predictions.append(prediction_df)

    if not predictions:
        print("No predictions generated.")
        return pd.Series(dtype=float), pd.Series(dtype=float)

    # Combine all individual forecasts
    predictions = pd.concat(predictions)
    predictions["time_to_delivery"] = predictions.eval("delivery_start - prediction_time")

    # merge with ground truth
    data = predictions.merge(
        target.to_frame("target"),
        left_on="delivery_start",
        right_index=True,
        how="inner",
    )
    
    # get metrics
    data["abs_error"] = data.eval("target - prediction").abs()
    data["sign_correct"] = np.sign(data.target) == np.sign(data.prediction)

    mae = data.groupby("time_to_delivery").abs_error.mean()
    sign_accuracy = data.groupby("time_to_delivery").sign_correct.mean()
    
    return mae, sign_accuracy

#MAIN

def main(
    task: Literal["predict", "backtest"],
    bidding_zone: BiddingZone,
    time_from: str | None = None,
):
    match task:
        case "predict":
            now = pd.Timestamp("now", tz="CET")
            forecast_horizon = get_forecast_horizon(now)
            
            print(f"--- Fast Analog Forecast (15m Aggregation) for {bidding_zone} at {now} ---")

            # Snap to last 15 min for candidate search to match grid
            effective_time = now.floor("15min")
            
            # 1) Get Analog Candidates
            print("1. Fetching candidate pool...")
            df_candidates = get_analog_candidates(
                past_days=PAST_DAYS,
                delta_hours=DELTA_HOURS,
                n_minutes=N_MINUTES_INPUT,
                now=effective_time,
                bidding_zone=bidding_zone
            )
            
            if df_candidates.empty:
                print("Warning: No candidates found. Check connectivity.")
                return

            # 2) Get & Aggregate Recent Data
            print("2. Fetching and aggregating live pattern...")
            
            # Fetch extra buffer to ensure we cover the full block
            raw_recent_pattern = get_recent_afrr_activation(
                bidding_zone=bidding_zone, 
                limit=N_MINUTES_INPUT + 60
            )
            
            # Slice strictly to the effective window
            raw_recent_pattern = raw_recent_pattern.loc[:effective_time].iloc[-N_MINUTES_INPUT:]
            
            if len(raw_recent_pattern) < N_MINUTES_INPUT:
                print(f"Error: Not enough live data. Got {len(raw_recent_pattern)} mins.")
                return
            
            # Reample to match candidate format (15min)
            recent_pattern_15min = raw_recent_pattern.resample('15min', closed='right', label='right').mean()
            recent_pattern_15min = recent_pattern_15min.ffill().bfill()
            
            # 3) Get Best Analogs
            print("3. Finding best matches...")
            best_analog_indices = get_best_analogs(
                k=K_BEST,
                df_analog_candidates=df_candidates,
                recent_afrr_activations=recent_pattern_15min
            )
            
            # 4) Get Future
            print("4. Retrieving future outcomes...")
            analogs_future = get_future_afrr_activation(
                idx=best_analog_indices,
                n_minutes=N_MINUTES_OUTPUT,
                bidding_zone=bidding_zone
            )
            
            if analogs_future.empty:
                 print("Error: Could not retrieve future data.")
                 return

            # 5) Predict
            print("5. Generating forecast...")
            prediction = predict_analogs(
                forecast_horizon=forecast_horizon,
                analogs_future=analogs_future
            )

            print(f"aFRR activation prediction for {bidding_zone}")
            print(prediction.to_string())
            
        case "backtest":
            assert time_from is not None, "time_from required"
            time_from = pd.Timestamp(time_from, tz="CET")
            
            print(f"Fetching backtest data starting from {time_from}...")
   
            activations = get_recent_afrr_activation(
                bidding_zone=bidding_zone, 
                time_from=time_from,
                limit=3000
            )
            
            mae, sign_accuracy = backtest(activations, bidding_zone)

            print(f"Backtest results for {bidding_zone}")
            print("Mean absolute error:")
            print(mae.to_string())
            print("Sign accuracy:")
            print(sign_accuracy.to_string())

if __name__ == "__main__":
    run(main)