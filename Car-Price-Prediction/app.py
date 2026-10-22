"""
AutoPrice — AI-powered car price prediction.

Uses the trained scikit-learn pipeline saved from the notebook
(ColumnTransformer[OneHotEncoder] -> LinearRegression), so the exact
training-time preprocessing is reused at prediction time.

Model input (exact order, verified from the pickle):
    Car Name, Year, Distance, Owner, Fuel, Drive, Type
"""

from __future__ import annotations

import html as _html
import json
import pickle
import re
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

# --------------------------------------------------------------------------- #
# Configuration (all values below were read from the notebook / pickle)
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).resolve().parent

MODEL_CANDIDATES = [
    BASE_DIR / "model" / "best_model.pkl",  # produced by train.py (preferred)
    BASE_DIR / "model" / "linearregression.pkl",
    BASE_DIR / "model" / "LinearRegressionModek.pkl",
    BASE_DIR / "model" / "LinearRegressionModel.pkl",
]
DATA_CANDIDATES = [
    BASE_DIR / "data" / "Cleaned_car.csv",
    BASE_DIR / "data" / "Cleaned_car.xls",  # NOTE: this file is really a CSV
    BASE_DIR / "data" / "cleanedcar.xls",
]

FEATURE_ORDER = ["Car Name", "Year", "Distance", "Owner", "Fuel", "Drive", "Type"]
CATEGORICAL = ["Car Name", "Fuel", "Drive", "Type"]  # same order as the OneHotEncoder

TRAIN_YEAR_MIN, TRAIN_YEAR_MAX = 2010, 2023
MAX_KM = 1_000_000
TYPICAL_MAX_KM = 200_000  # above this the data is very sparse

METRICS_PATH = BASE_DIR / "model" / "metrics.json"

# Shown when the original notebook model is used (no train.py run yet).
# From the notebook's random_state=483 split; MAE re-computed on that same split.
FALLBACK_INFO = {"name": "Linear Regression", "r2": 0.76, "mae": 88_000}

FUEL_LABELS = {"PETROL": "Petrol", "DIESEL": "Diesel", "CNG": "CNG", "LPG": "LPG"}
BODY_LABELS = {
    "HatchBack": "Hatchback",
    "Sedan": "Sedan",
    "SUV": "SUV",
    "Lux_SUV": "Luxury SUV",
    "Lux_sedan": "Luxury Sedan",
}
OWNER_LABELS = {1: "First owner", 2: "Second owner", 3: "Third owner", 4: "Fourth owner"}


class AppError(Exception):
    """An error whose message is safe to show to the user."""


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def format_inr(value: float, symbol: str = "₹") -> str:
    """Format a number with Indian digit grouping, e.g. 1245000 -> ₹12,45,000."""
    n = int(round(value))
    digits = str(abs(n))
    if len(digits) > 3:
        head, tail = digits[:-3], digits[-3:]
        head = re.sub(r"(\d)(?=(\d\d)+$)", r"\1,", head)
        digits = f"{head},{tail}"
    return f"{symbol}{'-' if n < 0 else ''}{digits}"


def esc(value: object) -> str:
    return _html.escape(str(value))


def render(markup: str) -> None:
    """Render raw HTML via st.markdown.

    Markdown treats indented lines / blank lines specially, so strip both.
    """
    cleaned = "\n".join(line.strip() for line in markup.splitlines() if line.strip())
    st.markdown(cleaned, unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Model + data loading
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def load_model():
    path = next((p for p in MODEL_CANDIDATES if p.exists()), None)
    if path is None:
        raise AppError("Model file not found. Place it at model/linearregression.pkl.")
    try:
        with open(path, "rb") as fh:  # only load pickles you trust
            model = pickle.load(fh)
    except Exception as exc:
        raise AppError(
            "The model file could not be loaded. It was saved with scikit-learn 1.9.1 — "
            "run `pip install -r requirements.txt` to install the matching version."
        ) from exc

    # Compatibility safety-net: pickles from scikit-learn 1.9 lack an attribute that
    # slightly older releases expect on ColumnTransformer. Harmless on 1.9.1.
    for _, step in getattr(model, "steps", []):
        if type(step).__name__ == "ColumnTransformer" and not hasattr(step, "force_int_remainder_cols"):
            step.force_int_remainder_cols = False

    expected = list(getattr(model, "feature_names_in_", []))
    if expected and expected != FEATURE_ORDER:
        raise AppError(
            "The model expects different input columns than this app provides: "
            f"{expected}."
        )
    return model


def model_info() -> dict:
    """Display name + metrics for whichever model file is active."""
    path = next((p for p in MODEL_CANDIDATES if p.exists()), None)
    if path is not None and path.name == "best_model.pkl" and METRICS_PATH.exists():
        try:
            m = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
            return {"name": m["model_name"], "r2": m["test"]["r2"], "mae": m["test"]["mae"]}
        except Exception:
            pass  # fall through to the defaults
    return dict(FALLBACK_INFO)


def get_categories(model) -> dict[str, list[str]]:
    """Read the exact category lists the encoder was trained with."""
    try:
        encoder = model.named_steps["columntransformer"].named_transformers_["onehotencoder"]
        return {name: list(cats) for name, cats in zip(CATEGORICAL, encoder.categories_)}
    except Exception as exc:
        raise AppError("The model's encoder has an unexpected structure.") from exc


@st.cache_data(show_spinner=False)
def load_dataset() -> pd.DataFrame | None:
    """Dataset is optional: it only powers the stats and the auto-fill of defaults."""
    path = next((p for p in DATA_CANDIDATES if p.exists()), None)
    if path is None:
        return None
    try:
        df = pd.read_csv(path, index_col=0)  # the ".xls" is actually CSV text
    except Exception:
        return None
    return df if set(FEATURE_ORDER + ["Price"]) <= set(df.columns) else None


@st.cache_data(show_spinner=False)
def car_defaults() -> dict[str, dict[str, str]]:
    """Most common fuel / transmission / body type for every car model."""
    df = load_dataset()
    if df is None:
        return {}
    out: dict[str, dict[str, str]] = {}
    for name, sub in df.groupby("Car Name"):
        out[name] = {c: sub[c].mode().iloc[0] for c in ("Fuel", "Drive", "Type")}
    return out


# --------------------------------------------------------------------------- #
# Prediction
# --------------------------------------------------------------------------- #
def validate(year: int | None, km: int | None) -> tuple[list[str], list[str]]:
    errors, notes = [], []
    this_year = date.today().year
    if year is None:
        errors.append("⚠️ Please enter the year of manufacture.")
    elif not TRAIN_YEAR_MIN <= year <= this_year:
        errors.append(f"⚠️ Please enter a valid year between {TRAIN_YEAR_MIN} and {this_year}.")
    elif year > TRAIN_YEAR_MAX:
        notes.append(f"ℹ️ The model was trained on cars from {TRAIN_YEAR_MIN}–{TRAIN_YEAR_MAX}; newer years are extrapolated.")
    if km is None:
        errors.append("⚠️ Please enter a valid number of kilometers.")
    elif km < 0:
        errors.append("⚠️ Kilometers driven cannot be negative.")
    elif km > MAX_KM:
        errors.append(f"⚠️ Kilometers driven looks too high (max {MAX_KM:,}).")
    elif km > TYPICAL_MAX_KM:
        notes.append("ℹ️ Very high mileage is rare in the training data, so this estimate may be less reliable.")
    return errors, notes


def predict_price(model, car: dict) -> float:
    row = {
        "Car Name": car["car"],
        "Year": float(car["year"]),  # trained as float64
        "Distance": int(car["km"]),
        "Owner": int(car["owner"]),
        "Fuel": car["fuel"],
        "Drive": car["drive"],
        "Type": car["body"],
    }
    frame = pd.DataFrame([[row[c] for c in FEATURE_ORDER]], columns=FEATURE_ORDER)
    try:
        price = float(model.predict(frame)[0])
    except Exception as exc:
        raise AppError("The model couldn't produce a prediction for these inputs.") from exc
    if price <= 0:
        raise AppError("This combination falls outside what the model can estimate reliably. Try different details.")
    return round(price / 1000) * 1000  # nearest ₹1,000 — avoids false precision


# --------------------------------------------------------------------------- #
# Styling
# --------------------------------------------------------------------------- #
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@300;400;500;600;700;800&display=swap');

:root{
  --bg:#0b0b0d; --card:rgba(255,255,255,.035); --card-solid:#141418;
  --line:rgba(255,255,255,.08); --text:#f3f1ec; --muted:#9a9aa3;
  --accent:#c9a96e; --accent-2:#e7d3a7;
}
html{scroll-behavior:smooth}
.stApp{
  background:
    radial-gradient(1200px 600px at 80% -10%, rgba(201,169,110,.10), transparent 60%),
    radial-gradient(900px 500px at -10% 10%, rgba(255,255,255,.04), transparent 60%),
    var(--bg);
  color:var(--text); font-family:'Manrope',system-ui,-apple-system,'Segoe UI',sans-serif;
}
.stApp *{font-family:'Manrope',system-ui,-apple-system,'Segoe UI',sans-serif}
#MainMenu, footer{visibility:hidden}
[data-testid="stHeader"]{background:transparent}
.block-container{max-width:1080px; padding:1.5rem 1.5rem 3rem}

/* Sidebar */
[data-testid="stSidebar"]{background:#0e0e11; border-right:1px solid var(--line)}
[data-testid="stSidebar"] *{color:var(--text)}
.side-k{color:var(--muted)!important; font-size:.72rem; letter-spacing:.14em; text-transform:uppercase; margin-top:1rem}
.side-v{font-weight:700; font-size:1.05rem}

/* Nav / hero */
.nav{display:flex; align-items:center; justify-content:space-between; padding:.4rem 0 1.2rem; border-bottom:1px solid var(--line)}
.brand{font-weight:800; font-size:1.25rem; letter-spacing:.01em}
.brand span{color:var(--accent)}
.tag{color:var(--muted); font-size:.85rem; letter-spacing:.04em}
.hero{position:relative; text-align:center; padding:4.5rem 1rem 3rem; overflow:hidden}
.hero .eyebrow{color:var(--accent); font-size:.78rem; letter-spacing:.32em; text-transform:uppercase; font-weight:600}
.hero h1{font-size:clamp(2.4rem,6vw,4.4rem); font-weight:800; line-height:1.05; letter-spacing:-.02em; margin:.8rem 0 1rem;
  background:linear-gradient(180deg,#fff 30%,#a9a9b1 120%); -webkit-background-clip:text; background-clip:text; color:transparent}
.hero p{color:var(--muted); max-width:560px; margin:0 auto 2rem; font-size:1.08rem; line-height:1.7}
.cta{display:inline-block; padding:.95rem 2.2rem; border-radius:999px; font-weight:700; text-decoration:none!important;
  color:#111!important; background:linear-gradient(135deg,var(--accent-2),var(--accent)); box-shadow:0 10px 30px rgba(201,169,110,.25);
  transition:transform .25s ease, box-shadow .25s ease}
.cta:hover{transform:translateY(-3px); box-shadow:0 16px 40px rgba(201,169,110,.35)}
.car-art{width:min(640px,92%); margin:2.6rem auto 0; display:block; opacity:.9}

/* Section headings */
.eyebrow{color:var(--accent); font-size:.74rem; letter-spacing:.28em; text-transform:uppercase; font-weight:700}
.section{margin-top:4rem}
.section h2, .card-title{font-size:1.6rem; font-weight:700; letter-spacing:-.01em; margin:.35rem 0 1.2rem; color:var(--text)}
.center{text-align:center}

/* Cards */
.card, .st-key-form_card{background:linear-gradient(160deg,rgba(255,255,255,.055),rgba(255,255,255,.02));
  border:1px solid var(--line); border-radius:22px; box-shadow:0 20px 50px rgba(0,0,0,.45);
  transition:transform .3s ease, border-color .3s ease}
.card{padding:1.4rem 1.5rem}
.card:hover{transform:translateY(-4px); border-color:rgba(201,169,110,.35)}
.st-key-form_card{padding:2rem 2rem 1.6rem; margin-top:3rem}
.st-key-form_card:hover{border-color:rgba(201,169,110,.25)}

/* Inputs */
[data-testid="stWidgetLabel"] p{color:var(--muted)!important; font-size:.78rem!important; letter-spacing:.1em; text-transform:uppercase; font-weight:600}
div[data-baseweb="select"] > div, div[data-baseweb="input"], div[data-baseweb="base-input"]{
  background:#101014!important; border:1px solid var(--line)!important; border-radius:12px!important; transition:border-color .2s ease, box-shadow .2s ease}
div[data-baseweb="select"] > div:hover, div[data-baseweb="input"]:hover{border-color:rgba(201,169,110,.5)!important}
div[data-baseweb="select"] > div:focus-within, div[data-baseweb="input"]:focus-within{border-color:var(--accent)!important; box-shadow:0 0 0 3px rgba(201,169,110,.15)}
.stNumberInput input, div[data-baseweb="select"] *{color:var(--text)!important; font-weight:500}
.stNumberInput button{background:transparent!important; border:none!important; color:var(--muted)!important}
div[data-baseweb="popover"] ul{background:#15151a!important; border:1px solid var(--line); border-radius:12px}
div[data-baseweb="popover"] li:hover{background:rgba(201,169,110,.12)!important}

/* Button */
.st-key-predict_btn button{
  margin-top:.8rem; padding:.85rem 1rem; border:none; border-radius:999px; font-weight:700; font-size:1.02rem;
  background:linear-gradient(135deg,var(--accent-2),var(--accent)); box-shadow:0 10px 30px rgba(201,169,110,.22);
  transition:transform .25s ease, box-shadow .25s ease, filter .25s ease}
.st-key-predict_btn button p{color:#111!important; font-weight:700}
.st-key-predict_btn button:hover{transform:translateY(-2px); box-shadow:0 16px 40px rgba(201,169,110,.35); filter:brightness(1.05)}
.st-key-predict_btn button:active{transform:translateY(0)}

/* Alerts */
[data-testid="stAlert"]{background:rgba(255,255,255,.04); border:1px solid var(--line); border-radius:14px; color:var(--text)}

/* Result */
@keyframes rise{from{opacity:0; transform:translateY(18px) scale(.98)} to{opacity:1; transform:none}}
.result{margin-top:2rem; text-align:center; padding:3rem 1.5rem; border-radius:26px; animation:rise .7s cubic-bezier(.2,.8,.2,1) both;
  border:1px solid rgba(201,169,110,.35); box-shadow:0 30px 70px rgba(0,0,0,.55), 0 0 80px rgba(201,169,110,.08);
  background:radial-gradient(600px 240px at 50% 0%, rgba(201,169,110,.16), transparent 70%), linear-gradient(160deg,rgba(255,255,255,.06),rgba(255,255,255,.015))}
.result .label{color:var(--accent); letter-spacing:.34em; font-size:.8rem; font-weight:700}
.result .price{font-size:clamp(3rem,9vw,6rem); font-weight:800; letter-spacing:-.03em; line-height:1.1; margin:.9rem 0 .6rem;
  background:linear-gradient(180deg,#fff,var(--accent-2)); -webkit-background-clip:text; background-clip:text; color:transparent}
.result .sub{color:var(--muted)}
.result .fine{color:#6f6f78; font-size:.8rem; margin-top:1.1rem}

/* Grids */
.grid{display:grid; gap:1rem; grid-template-columns:repeat(auto-fit,minmax(190px,1fr))}
.mini{padding:1rem 1.2rem; border-radius:16px; background:var(--card); border:1px solid var(--line); transition:transform .25s ease, border-color .25s ease}
.mini:hover{transform:translateY(-3px); border-color:rgba(201,169,110,.35)}
.mini .k{color:var(--muted); font-size:.72rem; letter-spacing:.14em; text-transform:uppercase}
.mini .v{font-weight:700; font-size:1.12rem; margin-top:.3rem}
.stat{text-align:center; padding:1.8rem 1rem}
.stat .big{font-size:2.3rem; font-weight:800; color:var(--accent-2); letter-spacing:-.02em}
.stat .cap{color:var(--muted); margin-top:.3rem}

/* How it works */
.steps{display:grid; gap:1rem; grid-template-columns:repeat(auto-fit,minmax(220px,1fr))}
.step{padding:1.6rem; text-align:center}
.step .num{font-size:2.4rem; font-weight:800; color:rgba(201,169,110,.55)}
.step .t{font-weight:700; margin-top:.2rem}
.step .d{color:var(--muted); font-size:.92rem; margin-top:.2rem}
.flow{max-width:760px; margin:1.8rem auto 0; padding:0; list-style:none; counter-reset:f}
.flow li{counter-increment:f; position:relative; padding:.7rem 0 .7rem 3rem; color:#cfcfd6; border-bottom:1px solid var(--line)}
.flow li::before{content:counter(f); position:absolute; left:0; top:.55rem; width:1.9rem; height:1.9rem; border-radius:50%;
  display:grid; place-items:center; font-weight:700; font-size:.85rem; color:#111; background:linear-gradient(135deg,var(--accent-2),var(--accent))}

/* Footer */
.footer{margin-top:5rem; padding:2.2rem 0 1rem; text-align:center; border-top:1px solid var(--line)}
.footer .b{font-weight:800; font-size:1.1rem}
.footer .s{color:var(--muted); margin:.3rem 0}
.footer .t{color:#6f6f78; font-size:.85rem}

@media (max-width:768px){
  .hero{padding:3rem .5rem 2rem}
  .st-key-form_card{padding:1.3rem 1.1rem}
  .nav .tag{display:none}
}
@media (prefers-reduced-motion:reduce){ *{animation:none!important; transition:none!important} }
</style>
"""

CAR_SVG = """
<svg class="car-art" viewBox="0 0 600 190" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
<defs>
<linearGradient id="g" x1="0" x2="1"><stop offset="0" stop-color="#c9a96e" stop-opacity="0"/><stop offset=".5" stop-color="#e7d3a7"/><stop offset="1" stop-color="#c9a96e" stop-opacity="0"/></linearGradient>
<linearGradient id="f" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#fff" stop-opacity=".07"/><stop offset="1" stop-color="#fff" stop-opacity="0"/></linearGradient>
</defs>
<path d="M40 138C40 118 68 110 108 106L184 68C206 54 238 48 288 48H368C410 48 440 68 470 98L518 106C548 110 560 124 560 138V148H40Z" fill="url(#f)" stroke="url(#g)" stroke-width="2"/>
<path d="M200 74C218 62 244 58 288 58H360C392 58 414 72 436 96H178Z" stroke="#c9a96e" stroke-opacity=".35" stroke-width="1.5"/>
<path d="M300 58V96" stroke="#c9a96e" stroke-opacity=".3" stroke-width="1.5"/>
<circle cx="150" cy="148" r="28" fill="#0b0b0d" stroke="#c9a96e" stroke-opacity=".8" stroke-width="2"/>
<circle cx="150" cy="148" r="11" stroke="#c9a96e" stroke-opacity=".5" stroke-width="1.5"/>
<circle cx="452" cy="148" r="28" fill="#0b0b0d" stroke="#c9a96e" stroke-opacity=".8" stroke-width="2"/>
<circle cx="452" cy="148" r="11" stroke="#c9a96e" stroke-opacity=".5" stroke-width="1.5"/>
<path d="M10 178H590" stroke="url(#g)" stroke-width="1.5"/>
</svg>
"""


# --------------------------------------------------------------------------- #
# UI sections
# --------------------------------------------------------------------------- #
def section_hero() -> None:
    render(
        f"""
        <div class="nav"><div class="brand">🚗 Auto<span>Price</span></div>
        <div class="tag">AI-Powered Car Price Prediction</div></div>
        <div class="hero">
        <div class="eyebrow">Machine Learning · Used Car Valuation</div>
        <h1>Know Your Car's Worth.</h1>
        <p>Predict the estimated market value of your car using machine learning —
        based on your vehicle's specifications.</p>
        <a class="cta" href="#car-details">Predict My Car</a>
        {CAR_SVG}
        </div>
        """
    )


def sync_defaults() -> None:
    """When the car model changes, pre-select its most common specs."""
    d = car_defaults().get(st.session_state.get("car"))
    if d:
        st.session_state.fuel, st.session_state.drive, st.session_state.body = d["Fuel"], d["Drive"], d["Type"]


def init_state(cats: dict[str, list[str]]) -> None:
    if "car" in st.session_state:
        return
    names = cats["Car Name"]
    st.session_state.car = next((n for n in ("Maruti Swift", "Hyundai Creta") if n in names), names[0])
    st.session_state.year = min(2019, date.today().year)
    st.session_state.km = 50_000
    st.session_state.owner = 1
    st.session_state.fuel = cats["Fuel"][0]
    st.session_state.drive = cats["Drive"][0]
    st.session_state.body = cats["Type"][0]
    sync_defaults()


def section_form(cats: dict[str, list[str]]) -> bool:
    """Returns True when the Predict button was clicked."""
    with st.container(key="form_card"):
        render(
            """
            <div id="car-details" class="center"><div class="eyebrow">Car Details</div>
            <div class="card-title">Tell us about your vehicle</div></div>
            """
        )
        st.selectbox("Car model", cats["Car Name"], key="car", on_change=sync_defaults)
        c1, c2 = st.columns(2)
        with c1:
            st.number_input("Year", min_value=TRAIN_YEAR_MIN, max_value=date.today().year, step=1, format="%d", key="year")
        with c2:
            st.number_input("Kms driven", min_value=0, max_value=MAX_KM, step=1000, format="%d", key="km")
        c3, c4 = st.columns(2)
        with c3:
            st.selectbox("Fuel type", cats["Fuel"], format_func=lambda v: FUEL_LABELS.get(v, v), key="fuel")
        with c4:
            st.selectbox("Transmission", cats["Drive"], key="drive")
        c5, c6 = st.columns(2)
        with c5:
            st.selectbox("Body type", cats["Type"], format_func=lambda v: BODY_LABELS.get(v, v), key="body")
        with c6:
            st.selectbox("Ownership", list(OWNER_LABELS), format_func=OWNER_LABELS.get, key="owner")
        return st.button("✨ Predict Price", key="predict_btn", use_container_width=True)


def section_result(res: dict, info: dict) -> None:
    render(
        f"""
        <div class="result">
        <div class="label">ESTIMATED VALUE</div>
        <div class="price">{format_inr(res["price"])}</div>
        <div class="sub">Based on your vehicle details</div>
        <div class="fine">Rounded to the nearest ₹1,000 · typical error on unseen cars ≈ {format_inr(round(info["mae"], -3))}</div>
        </div>
        """
    )
    c = res["car"]
    items = [
        ("Car", c["car"]),
        ("Year", c["year"]),
        ("Fuel", FUEL_LABELS.get(c["fuel"], c["fuel"])),
        ("Transmission", c["drive"]),
        ("Kilometers", format_inr(c["km"], symbol="")),
        ("Body type", BODY_LABELS.get(c["body"], c["body"])),
        ("Ownership", OWNER_LABELS[c["owner"]]),
    ]
    cards = "".join(f'<div class="mini"><div class="k">{esc(k)}</div><div class="v">{esc(v)}</div></div>' for k, v in items)
    render(f'<div class="section"><div class="eyebrow">Your Car</div><div class="card-title">Summary</div><div class="grid">{cards}</div></div>')


def section_stats(df: pd.DataFrame | None) -> None:
    rows = f"{len(df):,}" if df is not None else "Thousands of"
    stats = [(rows, "Cars Analyzed"), ("ML", "Powered Model"), ("Instant", "Prediction")]
    cards = "".join(f'<div class="card stat"><div class="big">{esc(b)}</div><div class="cap">{esc(c)}</div></div>' for b, c in stats)
    render(f'<div class="section"><div class="grid">{cards}</div></div>')


def section_how() -> None:
    render(
        """
        <div class="section center"><div class="eyebrow">How it works</div><h2>From specs to price in seconds</h2>
        <div class="steps">
        <div class="card step"><div class="num">01</div><div class="t">Enter</div><div class="d">Details</div></div>
        <div class="card step"><div class="num">02</div><div class="t">Analyze</div><div class="d">Features</div></div>
        <div class="card step"><div class="num">03</div><div class="t">Predict</div><div class="d">Price</div></div>
        </div>
        <ol class="flow" style="text-align:left">
        <li>Collects your vehicle information</li>
        <li>Processes it with the same preprocessing used during model training (one-hot encoding)</li>
        <li>Sends the processed data to the trained Linear Regression model</li>
        <li>Displays the estimated price</li>
        </ol></div>
        """
    )


def section_about(df: pd.DataFrame | None, n_models: int, info: dict) -> None:
    items = [
        ("Model", info["name"]),
        ("Purpose", "Car price prediction"),
        ("Input", "Car model, year, kms, fuel, transmission, body type, ownership"),
        ("Output", "Estimated selling price"),
        ("Test R²", f"{info['r2']:.2f}"),
        ("Mean abs. error", f"≈ {format_inr(round(info['mae'], -3))}"),
        ("Car models supported", str(n_models)),
    ]
    if df is not None:
        items.append(("Training rows", f"{len(df):,}"))
    cards = "".join(f'<div class="mini"><div class="k">{esc(k)}</div><div class="v" style="font-size:1rem">{esc(v)}</div></div>' for k, v in items)
    render(f'<div class="section"><div class="eyebrow">About the model</div><h2>Under the hood</h2><div class="grid">{cards}</div></div>')


def section_footer() -> None:
    render(
        """
        <div class="footer"><div class="b">AutoPrice</div>
        <div class="s">Machine Learning Car Price Prediction</div>
        <div class="t">Built with Python • Scikit-learn • Streamlit</div></div>
        """
    )


def sidebar(df: pd.DataFrame | None, n_models: int, info: dict) -> None:
    with st.sidebar:
        render('<div class="brand">🚗 Auto<span>Price</span></div>')
        render(
            f"""
            <div class="side-k">Model</div><div class="side-v">{esc(info["name"])}</div>
            <div class="side-k">Test R²</div><div class="side-v">{info["r2"]:.2f}</div>
            <div class="side-k">Car models</div><div class="side-v">{n_models}</div>
            <div class="side-k">Training rows</div><div class="side-v">{f"{len(df):,}" if df is not None else "—"}</div>
            """
        )
        st.caption("Estimates are indicative and based on historical listing data, not a formal valuation.")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    st.set_page_config(page_title="AutoPrice · Car Price Prediction", page_icon="🚗", layout="wide", initial_sidebar_state="collapsed")
    st.markdown(CSS, unsafe_allow_html=True)
    section_hero()

    try:
        model = load_model()
        cats = get_categories(model)
    except AppError as err:
        st.error(f"⚠️ {err}")
        st.stop()

    df = load_dataset()
    info = model_info()
    sidebar(df, len(cats["Car Name"]), info)
    init_state(cats)

    if section_form(cats):
        year, km = st.session_state.get("year"), st.session_state.get("km")
        errors, notes = validate(year, km)
        if errors:
            for e in errors:
                st.warning(e)
            st.session_state.pop("result", None)
        else:
            for n in notes:
                st.info(n)
            car = {k: st.session_state[k] for k in ("car", "year", "km", "fuel", "drive", "body", "owner")}
            try:
                with st.spinner("Analyzing your vehicle…"):
                    price = predict_price(model, car)
                st.session_state["result"] = {"price": price, "car": car}
            except AppError as err:
                st.session_state.pop("result", None)
                st.warning(f"⚠️ {err}")
            except Exception:
                st.session_state.pop("result", None)
                st.warning("⚠️ Something went wrong while predicting. Please check your inputs and try again.")

    if "result" in st.session_state:
        section_result(st.session_state["result"], info)

    section_stats(df)
    section_how()
    section_about(df, len(cats["Car Name"]), info)
    section_footer()


if __name__ == "__main__":
    main()
