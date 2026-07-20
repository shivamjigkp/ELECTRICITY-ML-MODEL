
# Electricity Demand Forecasting — End-to-End ML Project

Modular, production-style ML project (data ingestion → transformation →
training → Flask web app) that forecasts hourly electricity demand using
XGBoost, trained on 5 years of historical demand/weather/calendar data.

## Project structure


mlproject/
├── application.py                  # Flask app (main entry point)
├── app.py                          # alias entry point (some hosts expect app.py)
├── setup.py                        # makes src/ a pip-installable package
├── requirements.txt
├── .gitignore
├── artifacts/                      # generated at training time (git-ignored is optional)
│   ├── raw.csv
│   ├── train.csv
│   ├── test.csv
│   ├── preprocessor.pkl
│   └── model.pkl
├── notebook/
│   ├── data/Electricity_Demand_Dataset.csv
│   └── 1. EDA & MODEL TRAINING.ipynb
├── src/
│   ├── logger.py                   # custom logging setup, writes to logs/
│   ├── exception.py                # custom exception with file/line detail
│   ├── utils.py                    # save_object / load_object / evaluate_models
│   ├── components/
│   │   ├── data_ingestion.py       # reads raw CSV, splits by date into train/test
│   │   ├── data_transformation.py  # feature engineering (lag, rolling, calendar)
│   │   └── model_trainer.py        # trains XGBoost, saves artifacts/model.pkl
│   └── pipeline/
│       ├── train_pipeline.py       # orchestrates ingestion → transformation → training
│       └── predict_pipeline.py     # CustomData + PredictPipeline for inference
└── templates/
    ├── index.html                  # landing page
    └── home.html                   # prediction form + result


## How it works

1. **Data Ingestion** (`src/components/data_ingestion.py`) — reads the raw CSV,
   saves a copy to `artifacts/raw.csv`, and splits it by date (train: up to
   2023-12-31, test: 2024 onward) into `artifacts/train.csv` / `test.csv`.
2. **Data Transformation** (`src/components/data_transformation.py`) — builds
   calendar features (hour, dayofweek, month, quarter, weekofyear, is_weekend),
   and lag/rolling features (demand 24hr ago, demand 168hr ago, 24hr rolling
   mean/std). Saves the feature column order as `artifacts/preprocessor.pkl`.
3. **Model Training** (`src/components/model_trainer.py`) — trains an
   `XGBRegressor`, evaluates with RMSE/MAE/R², saves `artifacts/model.pkl`.
4. **Flask App** (`application.py`) — serves a form (`templates/home.html`)
   where a user enters weather + recent demand values, and returns a live
   prediction using `PredictPipeline`.

## Results (on this dataset)

| Metric | Value |
|---|---|
| RMSE | ~174.8 |
| MAE | ~123.3 |
| R² | ~0.985 |

## Run locally

```bash
git clone <your-repo-url>
cd mlproject
pip install -r requirements.txt

# train the model (creates everything in artifacts/)
python -m src.pipeline.train_pipeline

# run the web app
python application.py
# open http://127.0.0.1:5000
```

## Important note on this model

The model's strongest features are `Demand_lag_24hr` and `demand_lag_168hr`
(past actual demand values) — so it needs recent real demand history to
predict well, not just weather. That makes it best suited to **day-ahead /
batch forecasting** (a scheduled job that pulls recent demand + forecasts
the next day), rather than a zero-history real-time predictor. The web form
reflects this: it asks for recent demand values alongside temperature/humidity.

## Deployment (no AWS/Azure needed)

This Flask structure works well with **Render** or **Railway** (both have
free tiers, connect directly to a GitHub repo, no cloud console setup):

1. Push this project to GitHub.
2. On Render/Railway: "New Web Service" → connect the repo.
3. Build command: `pip install -r requirements.txt`
4. Start command: `gunicorn application:app` (add `gunicorn` to
   `requirements.txt` for production; the built-in Flask dev server used in
   `application.py` is fine for local testing only).
=======
