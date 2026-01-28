## Prediction Models

### 1. Mean Decay Model (`src/model_mean_decay.py`)

This is a parametric approach grounded in the physical nature of the power grid. Since aFRR activation is a corrective measure to restore frequency balance, it naturally tends to revert to zero once the imbalance is resolved.

#### Logic

The model forecasts a decay from the current value \( y_t \) towards zero using an exponential decay function:


#### The "Smart" Component (Dynamic λ)

Instead of using a fixed decay rate, the parameter \( \lambda \) is calculated dynamically for every prediction window:

- **Autocorrelation Analysis**  
  The model calculates the Lag-1 autocorrelation of the recent input window (e.g., last 60 minutes).

- **Dynamic Adjustment**
  - **High Autocorrelation**: Implies high system inertia or a persistent imbalance.  
    \( \lambda \) is set higher (slower decay).
  - **Low Autocorrelation**: Implies noise or fast-moving frequency corrections.  
    \( \lambda \) is set lower (fast reversion to zero).

---

### 2. Analog Forecasting Model (`src/analogs_model.py`)

This is a non-parametric approach based on the assumption that history repeats itself. The model searches in the past for market conditions similar to the current state and projects the future based on those historical outcomes.

#### Logic

- **Pattern Matching**  
  The model takes the last *N* minutes of realized activation (Input Window) and scans the past 30 days of data to find similar trajectories.

- **Search Optimization**  
  To ensure computational efficiency and relevance, the search is restricted to:
  - A specific time window around the current hour (e.g., ±2 hours) to capture intraday seasonality.
  - **15-minute Aggregation**: Candidates are matched based on 15-minute average blocks rather than raw minute data. This reduces dimensionality and focuses on the structural trend rather than noise.

- **Ensemble Prediction**  
  The model selects the *K* best matches (lowest Euclidean distance) and computes the weighted average of their future outcomes to generate the forecast.

***

##  Usage & Getting Started

Follow these steps to set up the environment and run the models.

### 1. Installation

Create and activate virtual environment

`python3 -m venv venv`

`source venv/bin/activate`

Install dependencies

`pip install -r requirements.txt`

### 2. Hyperparameter Optimization (Recommended First Step)
Before running predictions, generate the optimal configuration files for DK1 and DK2.

`python3 optimization/optimize_mean_decay.py`

Output:
This will create or update the following files:

`config/best_params_DK1.json`

`config/best_params_DK2.json`

### 3. Backtesting

Evaluate the models on historical data to see performance metrics (MAE and Sign Accuracy).

Backtest the Mean Decay Model:

`python3 src/model_mean_decay.py backtest DK1 --time-from "2026-01-28 00:00"`

Backtest the Analog Ensemble Model:

`python3 src/model_analogs.py backtest DK2 --time-from "2026-01-29 00:00"`


### 4. Live Prediction

Generate a forecast for the next 2 hours based on the latest real-time data from Energinet.

Predict using Mean Decay:

`python3 src/model_mean_decay.py predict DK1`

Predict using Analog Ensemble:

`python3 src/model_analogs.py predict DK1`
