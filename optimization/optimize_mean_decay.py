import pandas as pd
import numpy as np
import optuna
import os
import sys
import json
import warnings
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from src.model_mean_decay import backtest, get_recent_afrr_activation, BiddingZone

warnings.filterwarnings("ignore", category=RuntimeWarning)

#CONFIGURATION
TRAIN_DAYS = 1
N_TRIALS = 5

def run_optimization_for_zone(zone: BiddingZone):
    print("\n'======================================'")
    print(f"   STARTING OPTIMIZATION FOR {zone}")
    print("\n'======================================'")

    # 1. Fetch Data
    now = pd.Timestamp("now", tz="CET")
    start_time = now - pd.Timedelta(days=TRAIN_DAYS)
    
    print(f"Fetching training data for {zone}...")
    zone_activations = get_recent_afrr_activation(
        bidding_zone=zone,
        time_from=start_time,
        limit=50000 
    )
    
    if zone_activations.empty:
        print(f"Error: No data fetched for {zone}. Skipping.")
        return

    #2. Define objective Function
    def objective(trial):
        # Define the search space for our hyperparameters
        min_lambda = trial.suggest_float("min_lambda", 0.70, 0.95)
        max_lambda = trial.suggest_float("max_lambda", 0.95, 0.999)
        input_minutes = trial.suggest_int("n_minutes_input", 30, 120, step=15)
        
        # Sanity check: min decay shouldn't be higher than max
        if min_lambda >= max_lambda:
            raise optuna.TrialPruned()

        # Build the config dict to pass to the model
        params = {
            "MIN_LAMBDA": min_lambda,
            "MAX_LAMBDA": max_lambda,
            "N_MINUTES_INPUT": input_minutes,
            "DEFAULT_LAMBDA": 0.98  # Backoup parameter if optimization fails
        }
        
        # Run  backtest with  specific params
        mae, _ = backtest(zone_activations, params=params)
        
        # If backtest fails return infinite error
        if mae.empty:
            return float("inf")

        return mae.mean()

    # 3. Run optuna to minimize mae
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=N_TRIALS)
    
    print(f"\nBest Parameters for {zone}:")
    print(study.best_params)
    
    #4. Save results
    # save the best params in JSON file 
    project_root = Path(__file__).parent.parent
    config_dir = project_root / "config"
    config_dir.mkdir(exist_ok=True)
    
    filename = config_dir / f"best_params_{zone}.json"
    
    with open(filename, "w") as f:
        json.dump(study.best_params, f, indent=4)
        
    print(f"Saved config to {filename}")

if __name__ == "__main__":
    # Run for both DK1 and DK2
    for zone in [BiddingZone.DK1, BiddingZone.DK2]:
        run_optimization_for_zone(zone)