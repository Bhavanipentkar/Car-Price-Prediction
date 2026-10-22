# 🚗 AutoPrice — AI-Powered Car Price Prediction

A modern Streamlit web app that estimates the market value of a used car in India using a trained scikit-learn Linear Regression pipeline.

## Screenshots

> Run the app, take screenshots, and save them in a `screenshots/` folder.

| Hero & form | Prediction result |
| --- | --- |
| `screenshots/home.png` | `screenshots/result.png` |

## Project overview

AutoPrice takes a handful of vehicle details (model, year, kilometers driven, fuel, transmission, body type, ownership) and returns an estimated price in ₹, formatted in the Indian numbering system (e.g. `₹5,36,000`). The app loads the **already-trained** pipeline from the notebook — nothing is retrained at runtime.

## Features

- Premium dark UI with custom CSS (cards, gradients, hover effects, animated result)
- Reproducible training script with honest cross-validated model comparison
- EDA notebook with embedded charts
- Dropdowns limited to the categories the model was trained on (the encoder rejects unseen values)
- Smart defaults: choosing a car pre-selects its most common fuel, transmission and body type
- Input validation with friendly messages (missing/negative kms, invalid year, model or prediction errors)
- Indian currency formatting; result rounded to the nearest ₹1,000
- Live dataset statistics, "How it works" and "About the model" sections
- Responsive layout; works offline (no external images; the font falls back gracefully)

## Dataset

- Source: Cars24 listings (`cars_24_combined.csv`), cleaned in the notebook and saved as `data/Cleaned_car.csv`
- **8,014 rows**, 126 car models, years 2010–2023
- Columns: `Car Name, Year, Distance, Owner, Fuel, Drive, Type, Price`
- Cleaning: dropped `Unnamed: 0` and `Location`, removed the one row with a missing car name/year

> The original file was named `Cleaned_car.xls` but is actually CSV text, so it is stored here with a `.csv` extension.

## Machine learning approach

| Step | Detail |
| --- | --- |
| Target | `Price` (₹) |
| Categorical features | `Car Name` (126), `Fuel` (4), `Drive` (2), `Type` (5) → one-hot encoded |
| Numeric features | `Year`, `Distance`, `Owner` → passed through unchanged (no scaling needed) |
| Pipeline | `ColumnTransformer(OneHotEncoder) → estimator` (140 input columns after encoding) |
| Evaluation | One fixed 80/20 split (`random_state=42`) + 5-fold cross-validation on the training part |

`train.py` compares six models, selects the winner by **cross-validated MAE**, scores it **once** on the untouched test set, then refits it on all data and saves `model/best_model.pkl` and `model/metrics.json`. The app automatically prefers these files and falls back to the original notebook model (`linearregression.pkl`) if they don't exist.

## Model

Cross-validation results on the 6,411 training rows (output of `python train.py`; numbers can differ slightly across library versions):

| Model | CV R² | CV MAE (₹) |
| --- | --- | --- |
| Linear Regression (original) | 0.796 | 77,993 |
| Ridge Regression | 0.793 | 78,708 |
| **Linear Regression (log price)** ← selected | **0.802** | **73,542** |
| Random Forest | 0.795 | 75,379 |
| Gradient Boosting | 0.784 | 78,276 |
| Gradient Boosting (log price) | 0.784 | 76,870 |

Held-out test set: **R² ≈ 0.80, MAE ≈ ₹74,000** for the selected model, versus R² ≈ 0.77, MAE ≈ ₹81,000 for the original linear regression on the same split.

**What this shows**

- All six models land within about 0.02 R² of each other. Tree-based models did **not** beat the linear model; a further experiment with target-encoded car names also plateaued near 0.79.
- The likely reason is the data, not the algorithm: there is no trim/variant, location or condition column. A simple, explainable model is therefore a sensible choice.
- The original notebook's R² of 0.76 came from picking the best of 1,000 random splits (`random_state=483`), which is optimistic selection. The cross-validated numbers above are the trustworthy ones.

## Technologies used

Python · pandas · scikit-learn · Streamlit · matplotlib · HTML/CSS

## Project structure

```text
Car-Price-Prediction/
├── .streamlit/config.toml        # dark theme defaults
├── data/Cleaned_car.csv
├── model/
│   ├── linearregression.pkl      # original notebook model (fallback)
│   ├── best_model.pkl            # created by train.py
│   └── metrics.json              # created by train.py
├── notebook/
│   ├── car_predicting.ipynb      # original modelling notebook
│   └── eda.ipynb                 # exploratory data analysis with charts
├── app.py                        # Streamlit web app
├── train.py                      # reproducible training + evaluation
├── requirements.txt
├── requirements-dev.txt          # extras for the EDA notebook
├── README.md
└── .gitignore
```

## Installation

Requires **Python 3.11+**. The original model was pickled with scikit-learn **1.9.1**, so that exact version is pinned.

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux
pip install -r requirements.txt
```

## Running locally

```bash
python train.py          # optional but recommended: trains, evaluates, saves best_model.pkl (about a minute)
streamlit run app.py     # then open http://localhost:8501
```

To explore the data: `pip install -r requirements-dev.txt`, then `jupyter notebook` and open `notebook/eda.ipynb`.

## How prediction works

1. The form collects the 7 model inputs.
2. They are assembled into a one-row DataFrame with the **exact column order** the model was trained on: `Car Name, Year, Distance, Owner, Fuel, Drive, Type`.
3. The pipeline's own `ColumnTransformer` applies the training-time one-hot encoding.
4. The trained model outputs a price, which is rounded and shown in Indian format.

## Limitations

- Linear Regression can extrapolate poorly for extreme inputs (very high mileage, years outside 2010–2023); the app warns or declines in those cases.
- Estimates reflect historical Cars24 listing prices, not a formal valuation.
- Only load pickle files you trust.

## Future improvements

- Collect richer data (trim/variant, city, condition) — the main accuracy bottleneck
- Show a prediction interval instead of a single number
- Hyperparameter tuning and a larger model search once more data is available
- Add price-vs-age and price-vs-mileage charts inside the app
- Deploy on Streamlit Community Cloud
