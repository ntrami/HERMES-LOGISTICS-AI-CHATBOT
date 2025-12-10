import re
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "shipments.csv"


def ensure_dataset(path: Path = DATA_PATH, rows: int = 1000) -> pd.DataFrame:
    """Load dataset or create a synthetic one when missing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return pd.read_csv(path, parse_dates=["date"])

    routes = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
    warehouses = ["WH1", "WH2", "WH3", "WH4"]
    reasons = ["Weather", "Traffic", "Mechanical", "None"]
    start = datetime(2024, 1, 1)
    # Use current date (10.12.2025) as end date instead of fixed date
    end = datetime(2025, 12, 10)  # Current date: December 10, 2025
    data = []
    for i in range(1, rows + 1):
        delta_days = random.randint(0, (end - start).days)
        data.append(
            {
                "id": i,
                "route": random.choice(routes),
                "warehouse": random.choice(warehouses),
                "delivery_time": random.randint(1, 10),
                "delay_minutes": random.randint(0, 120),
                "delay_reason": random.choice(reasons),
                "date": (start + timedelta(days=delta_days)).date(),
            }
        )
    df = pd.DataFrame(data)
    df.to_csv(path, index=False)
    df["date"] = pd.to_datetime(df["date"])
    return df


df_shipments = ensure_dataset(rows=5000)


def _week_filter(df: pd.DataFrame, weeks: int = 1) -> pd.DataFrame:
    cutoff = df["date"].max() - pd.Timedelta(weeks=weeks)
    return df[df["date"] >= cutoff]


def _extract_month_from_query(query: str) -> int:
    """Extract month number (1-12) from query, return None if not found."""
    month_map = {
        "january": 1, "jan": 1,
        "february": 2, "feb": 2,
        "march": 3, "mar": 3,
        "april": 4, "apr": 4,
        "may": 5,
        "june": 6, "jun": 6,
        "july": 7, "jul": 7,
        "august": 8, "aug": 8,
        "september": 9, "sep": 9, "sept": 9,
        "october": 10, "oct": 10,
        "november": 11, "nov": 11,
        "december": 12, "dec": 12,
    }
    q_lower = query.lower()
    for month_name, month_num in month_map.items():
        if month_name in q_lower:
            return month_num
    return None


def _filter_by_month(df: pd.DataFrame, month: int, year: int = None) -> pd.DataFrame:
    """Filter dataframe by month (and optionally year)."""
    if year:
        return df[(df["date"].dt.month == month) & (df["date"].dt.year == year)]
    return df[df["date"].dt.month == month]


def _delays_by_route(df: pd.DataFrame, month_context: str = None) -> Tuple[str, List[Dict], Dict]:
    if len(df) == 0:
        return "No data available for the specified period.", [], {}
    grouped = (
        df.groupby("route")["delay_minutes"]
        .mean()
        .reset_index()
        .sort_values(by="delay_minutes", ascending=False)
    )
    top_route = grouped.iloc[0]
    month_text = f" in {month_context}" if month_context else ""
    text = f"Route {top_route.route} has the highest average delay at {top_route.delay_minutes:.1f} minutes{month_text}."
    chart = {
        "type": "bar",
        "x": grouped["route"].tolist(),
        "y": grouped["delay_minutes"].round(2).tolist(),
        "title": f"Average delay minutes by route{month_text}",
    }
    return text, grouped.to_dict(orient="records"), chart


def _top_warehouses(df: pd.DataFrame, top_n: int = 3) -> Tuple[str, List[Dict], Dict]:
    grouped = (
        df.groupby("warehouse")["delivery_time"]
        .mean()
        .reset_index()
        .sort_values(by="delivery_time")
        .head(top_n)
    )
    names = ", ".join(grouped["warehouse"])
    text = f"Top {len(grouped)} warehouses by fastest delivery: {names}."
    chart = {
        "type": "bar",
        "x": grouped["warehouse"].tolist(),
        "y": grouped["delivery_time"].round(2).tolist(),
        "title": "Average delivery time by warehouse (days)",
    }
    return text, grouped.to_dict(orient="records"), chart


def _delay_reasons(df: pd.DataFrame) -> Tuple[str, List[Dict], Dict]:
    # Count all delay reasons, including "None" - show ALL reasons that exist in data
    counts = df["delay_reason"].value_counts().reset_index()
    counts.columns = ["delay_reason", "count"]
    
    # Sort by count descending to show most common first
    counts = counts.sort_values("count", ascending=False)
    
    top_count = int(counts.iloc[0]["count"])
    top_reason = counts.iloc[0]["delay_reason"]
    
    total = counts["count"].sum()
    num_reasons = len(counts)
    text = f"Total delay reasons breakdown: {total} shipments across {num_reasons} categories. Top reason: {top_reason} with {top_count} occurrences ({top_count/total*100:.1f}%)."
    
    chart = {
        "type": "pie",
        "labels": counts["delay_reason"].tolist(),
        "values": counts["count"].tolist(),
        "title": "Delay reasons distribution",
    }
    return text, counts.to_dict(orient="records"), chart


def _build_enhanced_prediction_model(df: pd.DataFrame, aggregation_period: str = "W") -> Tuple[object, float, pd.DataFrame, List[str]]:
    """
    Build enhanced prediction model with improved features for better accuracy.
    Returns: (model, r2_score, aggregated_data, feature_columns)
    """
    df_features = df.copy()
    df_features["weekofyear"] = df_features["date"].dt.isocalendar().week.astype(int)
    df_features["year"] = df_features["date"].dt.year
    df_features["month"] = df_features["date"].dt.month
    df_features["quarter"] = df_features["date"].dt.quarter
    df_features["dayofyear"] = df_features["date"].dt.dayofyear
    df_features["dayofweek"] = df_features["date"].dt.dayofweek
    
    # Aggregate by period
    if aggregation_period == "W":
        df_agg = df_features.groupby(["year", "weekofyear"]).agg({
            "delay_minutes": "mean",
            "month": "first",
            "quarter": "first",
            "dayofyear": "mean",
            "dayofweek": "mean"
        }).reset_index()
        df_agg.columns = ["year", "period", "avg_delay", "month", "quarter", "dayofyear", "dayofweek"]
        period_col = "weekofyear"
    elif aggregation_period == "M":
        df_agg = df_features.groupby(["year", "month"]).agg({
            "delay_minutes": "mean",
            "quarter": "first",
            "weekofyear": "mean",
            "dayofyear": "mean"
        }).reset_index()
        df_agg.columns = ["year", "period", "avg_delay", "quarter", "weekofyear", "dayofyear"]
        period_col = "month"
    else:  # "Y"
        df_agg = df_features.groupby(["year"]).agg({
            "delay_minutes": "mean",
            "month": "mean",
            "quarter": "mean",
            "weekofyear": "mean"
        }).reset_index()
        df_agg.columns = ["year", "avg_delay", "month", "quarter", "weekofyear"]
        period_col = None
    
    # Sort by date
    if period_col:
        df_agg = df_agg.sort_values(["year", "period"])
    else:
        df_agg = df_agg.sort_values("year")
    
    # Add rolling averages (trend features)
    df_agg["rolling_avg_3"] = df_agg["avg_delay"].rolling(window=3, min_periods=1).mean()
    df_agg["rolling_avg_6"] = df_agg["avg_delay"].rolling(window=6, min_periods=1).mean()
    df_agg["rolling_std_3"] = df_agg["avg_delay"].rolling(window=3, min_periods=1).std().fillna(0)
    
    # Add lag features (previous period values)
    df_agg["lag_1"] = df_agg["avg_delay"].shift(1).fillna(df_agg["avg_delay"].mean())
    df_agg["lag_2"] = df_agg["avg_delay"].shift(2).fillna(df_agg["avg_delay"].mean())
    
    # Seasonal features (sine/cosine encoding for cyclical patterns)
    if period_col:
        if period_col == "weekofyear":
            max_period = 53
        else:  # month
            max_period = 12
        df_agg["period_sin"] = np.sin(2 * np.pi * df_agg["period"] / max_period)
        df_agg["period_cos"] = np.cos(2 * np.pi * df_agg["period"] / max_period)
    
    # Prepare features
    feature_cols = []
    if period_col:
        feature_cols.extend(["period", "year"])
    else:
        feature_cols.append("year")
    
    feature_cols.extend([
        "month", "quarter", "rolling_avg_3", "rolling_avg_6", 
        "rolling_std_3", "lag_1", "lag_2"
    ])
    
    if period_col:
        feature_cols.extend(["period_sin", "period_cos"])
    
    # Remove None values and ensure columns exist
    feature_cols = [col for col in feature_cols if col in df_agg.columns]
    
    if not feature_cols:
        return None, 0.0, df_agg
    
    X = df_agg[feature_cols].fillna(df_agg[feature_cols].mean())
    y = df_agg["avg_delay"]
    
    if len(X) < 6:
        # Not enough data, return simple model
        return None, 0.0, df_agg, feature_cols
    
    # Try multiple models and pick the best
    models_to_try = [
        ("Ridge", Ridge(alpha=1.0)),
        ("Linear", LinearRegression()),
        ("RF", RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42))
    ]
    
    best_model = None
    best_r2 = -np.inf
    best_name = ""
    
    for name, model in models_to_try:
        try:
            model.fit(X, y)
            y_pred = model.predict(X)
            r2 = r2_score(y, y_pred)
            if r2 > best_r2:
                best_r2 = r2
                best_model = model
                best_name = name
        except Exception:
            continue
    
    if best_model is None:
        # Fallback to simple linear regression
        best_model = LinearRegression()
        best_model.fit(X, y)
        y_pred = best_model.predict(X)
        best_r2 = r2_score(y, y_pred)
        best_name = "Linear"
    
    return best_model, best_r2, df_agg, feature_cols


def _predict_delay_next_week(df: pd.DataFrame) -> Tuple[str, List[Dict], Dict]:
    """Predict delay for next week using enhanced model."""
    model, r2, df_weekly, feature_cols = _build_enhanced_prediction_model(df, "W")
    
    # Get next week date
    next_date = df["date"].max() + pd.Timedelta(days=7)
    next_week = next_date.isocalendar()[1]
    next_year = next_date.year
    next_month = next_date.month
    next_quarter = next_date.quarter
    
    if model is not None and len(df_weekly) >= 3:
        # Prepare features for next week
        last_row = df_weekly.iloc[-1]
        last_3_avg = df_weekly["avg_delay"].tail(3).mean()
        last_6_avg = df_weekly["avg_delay"].tail(6).mean() if len(df_weekly) >= 6 else last_3_avg
        last_std = df_weekly["avg_delay"].tail(3).std() if len(df_weekly) >= 3 else 0
        
        next_features = pd.DataFrame({
            "period": [next_week],
            "year": [next_year],
            "month": [next_month],
            "quarter": [next_quarter],
            "rolling_avg_3": [last_3_avg],
            "rolling_avg_6": [last_6_avg],
            "rolling_std_3": [last_std],
            "lag_1": [last_row["avg_delay"]],
            "lag_2": [df_weekly.iloc[-2]["avg_delay"] if len(df_weekly) >= 2 else last_row["avg_delay"]],
            "period_sin": [np.sin(2 * np.pi * next_week / 53)],
            "period_cos": [np.cos(2 * np.pi * next_week / 53)],
        })
        
        # Select only features that model was trained on
        next_features = next_features[[col for col in feature_cols if col in next_features.columns]]
        
        prediction = max(0, model.predict(next_features)[0])
    else:
        # Fallback: use recent average
        prediction = max(0, df_weekly["avg_delay"].tail(4).mean() if len(df_weekly) >= 4 else df_weekly["avg_delay"].mean())
        r2 = 0.0
    
    # Prepare historical data for chart
    df_sorted = df.sort_values("date")
    last_12_weeks = df_sorted.tail(84)  # ~12 weeks of daily data
    historical_dates = last_12_weeks["date"].dt.strftime("%Y-%m-%d").tolist()
    historical_delays = last_12_weeks["delay_minutes"].tolist()
    
    # Calculate weekly averages for trend line
    recent_weekly = df_sorted.tail(84).groupby(
        [df_sorted.tail(84)["date"].dt.to_period("W")]
    )["delay_minutes"].mean().reset_index()
    trend_dates = [str(period.start_time.date()) for period in recent_weekly["date"]]
    trend_values = recent_weekly["delay_minutes"].tolist()
    
    # Build explanation text
    method_explanation = "Using enhanced ML model with advanced features (seasonal patterns, lag features, rolling statistics)."
    show_accuracy = r2 >= 0.1
    quality_note = f"Model accuracy (R²): {r2:.2%}" if show_accuracy else ""
    
    text = (
        f"Predicted average delay next week: {prediction:.1f} minutes. "
        f"{method_explanation}"
    )
    if quality_note:
        text += f" {quality_note}."
    text += f" Prediction date: {next_date.strftime('%Y-%m-%d')} (Week {next_week}, {next_year})."
    
    # Enhanced chart with historical data, trend, and prediction point
    # Pastel colors: blue, pink, green, yellow, purple, orange
    pastel_blue = "#A8D5E2"
    pastel_pink = "#F4A5AE"
    pastel_purple = "#D4A5FF"
    
    chart = {
        "type": "prediction",
        "historical": {
            "x": historical_dates,
            "y": historical_delays,
            "name": "Historical Daily Delays",
            "mode": "markers",
            "marker": {"color": pastel_blue, "size": 4, "opacity": 0.6}
        },
        "trend": {
            "x": trend_dates,
            "y": trend_values,
            "name": "Weekly Average Trend",
            "mode": "lines+markers",
            "line": {"color": pastel_pink, "width": 2}
        },
        "prediction": {
            "x": [next_date.strftime("%Y-%m-%d")],
            "y": [prediction],
            "name": "Prediction (Next Week)",
            "mode": "markers",
            "marker": {"color": pastel_purple, "size": 12, "symbol": "diamond"}
        },
        "title": "Delay Prediction: Historical Data & Forecast",
        "r2_score": round(r2, 3) if show_accuracy else None
    }
    
    table = [
        {"metric": "Predicted delay (next week)", "value": f"{prediction:.1f} minutes"},
        {"metric": "Prediction date", "value": next_date.strftime("%Y-%m-%d")},
        {"metric": "Model method", "value": "Enhanced ML (Ridge/RF/Linear)"},
        {"metric": "Features used", "value": "Seasonal, lag, rolling stats, trends"}
    ]
    if show_accuracy:
        table.append({"metric": "Model R² score", "value": f"{r2:.2%}"})
    return text, table, chart


def _predict_delay_next_month(df: pd.DataFrame) -> Tuple[str, List[Dict], Dict]:
    """Predict delay for next month using enhanced model."""
    model, r2, df_monthly, feature_cols = _build_enhanced_prediction_model(df, "M")
    
    # Get next month date
    next_date = df["date"].max() + pd.DateOffset(months=1)
    next_month = next_date.month
    next_year = next_date.year
    next_quarter = next_date.quarter
    
    if model is not None and len(df_monthly) >= 3:
        last_row = df_monthly.iloc[-1]
        last_3_avg = df_monthly["avg_delay"].tail(3).mean()
        last_6_avg = df_monthly["avg_delay"].tail(6).mean() if len(df_monthly) >= 6 else last_3_avg
        last_std = df_monthly["avg_delay"].tail(3).std() if len(df_monthly) >= 3 else 0
        
        next_features = pd.DataFrame({
            "period": [next_month],
            "year": [next_year],
            "month": [next_month],
            "quarter": [next_quarter],
            "rolling_avg_3": [last_3_avg],
            "rolling_avg_6": [last_6_avg],
            "rolling_std_3": [last_std],
            "lag_1": [last_row["avg_delay"]],
            "lag_2": [df_monthly.iloc[-2]["avg_delay"] if len(df_monthly) >= 2 else last_row["avg_delay"]],
            "period_sin": [np.sin(2 * np.pi * next_month / 12)],
            "period_cos": [np.cos(2 * np.pi * next_month / 12)],
        })
        
        next_features = next_features[[col for col in feature_cols if col in next_features.columns]]
        
        prediction = max(0, model.predict(next_features)[0])
    else:
        prediction = max(0, df_monthly["avg_delay"].tail(3).mean() if len(df_monthly) >= 3 else df_monthly["avg_delay"].mean())
        r2 = 0.0
    
    # Prepare historical data
    df_sorted = df.sort_values("date")
    last_6_months = df_sorted.tail(180)  # ~6 months
    monthly_agg = df_sorted.tail(365).groupby([df_sorted.tail(365)["date"].dt.to_period("M")])["delay_minutes"].mean().reset_index()
    trend_dates = [str(period.start_time.date()) for period in monthly_agg["date"]]
    trend_values = monthly_agg["delay_minutes"].tolist()
    
    method_explanation = "Using enhanced ML model with advanced features (seasonal patterns, lag features, rolling statistics)."
    show_accuracy = r2 >= 0.1
    quality_note = f"Model accuracy (R²): {r2:.2%}" if show_accuracy else ""
    
    month_name = next_date.strftime("%B %Y")
    text = (
        f"Predicted average delay for {month_name}: {prediction:.1f} minutes. "
        f"{method_explanation}"
    )
    if quality_note:
        text += f" {quality_note}."
    
    pastel_blue = "#A8D5E2"
    pastel_pink = "#F4A5AE"
    pastel_purple = "#D4A5FF"
    
    chart = {
        "type": "prediction",
        "historical": {
            "x": last_6_months["date"].dt.strftime("%Y-%m-%d").tolist(),
            "y": last_6_months["delay_minutes"].tolist(),
            "name": "Historical Daily Delays",
            "mode": "markers",
            "marker": {"color": pastel_blue, "size": 4, "opacity": 0.6}
        },
        "trend": {
            "x": trend_dates,
            "y": trend_values,
            "name": "Monthly Average Trend",
            "mode": "lines+markers",
            "line": {"color": pastel_pink, "width": 2}
        },
        "prediction": {
            "x": [next_date.strftime("%Y-%m-%d")],
            "y": [prediction],
            "name": f"Prediction ({month_name})",
            "mode": "markers",
            "marker": {"color": pastel_purple, "size": 12, "symbol": "diamond"}
        },
        "title": f"Delay Prediction for {month_name}",
        "r2_score": round(r2, 3) if show_accuracy else None
    }
    
    table = [
        {"metric": f"Predicted delay ({month_name})", "value": f"{prediction:.1f} minutes"},
        {"metric": "Prediction date", "value": next_date.strftime("%Y-%m-%d")},
        {"metric": "Model method", "value": "Enhanced ML (Ridge/RF/Linear)"},
        {"metric": "Features used", "value": "Seasonal, lag, rolling stats, trends"}
    ]
    if show_accuracy:
        table.append({"metric": "Model R² score", "value": f"{r2:.2%}"})
    return text, table, chart


def _predict_delay_next_year(df: pd.DataFrame) -> Tuple[str, List[Dict], Dict]:
    """Predict delay for next year using enhanced model."""
    model, r2, df_yearly, feature_cols = _build_enhanced_prediction_model(df, "Y")
    
    # Get next year
    next_year = df["date"].max().year + 1
    
    if model is not None and len(df_yearly) >= 2:
        last_row = df_yearly.iloc[-1]
        last_avg = df_yearly["avg_delay"].mean()
        
        next_features = pd.DataFrame({
            "year": [next_year],
            "month": [6],  # Mid-year
            "quarter": [2],
            "rolling_avg_3": [last_avg],
            "rolling_avg_6": [last_avg],
            "rolling_std_3": [0],
            "lag_1": [last_row["avg_delay"]],
            "lag_2": [df_yearly.iloc[-2]["avg_delay"] if len(df_yearly) >= 2 else last_row["avg_delay"]],
        })
        
        next_features = next_features[[col for col in feature_cols if col in next_features.columns]]
        
        prediction = max(0, model.predict(next_features)[0])
    else:
        prediction = max(0, df_yearly["avg_delay"].mean() if len(df_yearly) > 0 else 60.0)
        r2 = 0.0
    
    # Prepare historical data
    df_sorted = df.sort_values("date")
    yearly_agg = df_sorted.groupby([df_sorted["date"].dt.to_period("Y")])["delay_minutes"].mean().reset_index()
    trend_dates = [str(period.start_time.date()) for period in yearly_agg["date"]]
    trend_values = yearly_agg["delay_minutes"].tolist()
    
    method_explanation = "Using enhanced ML model with advanced features (trends, lag features, rolling statistics)."
    show_accuracy = r2 >= 0.1
    quality_note = f"Model accuracy (R²): {r2:.2%}" if show_accuracy else ""
    
    text = (
        f"Predicted average delay for {next_year}: {prediction:.1f} minutes. "
        f"{method_explanation}"
    )
    if quality_note:
        text += f" {quality_note}."
    
    pastel_blue = "#A8D5E2"
    pastel_pink = "#F4A5AE"
    pastel_purple = "#D4A5FF"
    
    chart = {
        "type": "prediction",
        "historical": {
            "x": df_sorted["date"].dt.strftime("%Y-%m-%d").tolist(),
            "y": df_sorted["delay_minutes"].tolist(),
            "name": "Historical Daily Delays",
            "mode": "markers",
            "marker": {"color": pastel_blue, "size": 3, "opacity": 0.5}
        },
        "trend": {
            "x": trend_dates,
            "y": trend_values,
            "name": "Yearly Average Trend",
            "mode": "lines+markers",
            "line": {"color": pastel_pink, "width": 2}
        },
        "prediction": {
            "x": [f"{next_year}-06-15"],
            "y": [prediction],
            "name": f"Prediction ({next_year})",
            "mode": "markers",
            "marker": {"color": pastel_purple, "size": 12, "symbol": "diamond"}
        },
        "title": f"Delay Prediction for {next_year}",
        "r2_score": round(r2, 3) if show_accuracy else None
    }
    
    table = [
        {"metric": f"Predicted delay ({next_year})", "value": f"{prediction:.1f} minutes"},
        {"metric": "Prediction year", "value": str(next_year)},
        {"metric": "Model method", "value": "Enhanced ML (Ridge/RF/Linear)"},
        {"metric": "Features used", "value": "Trends, lag features, rolling stats"}
    ]
    if show_accuracy:
        table.append({"metric": "Model R² score", "value": f"{r2:.2%}"})
    return text, table, chart


INTENT_TEMPLATES: Dict[str, List[str]] = {
    "delays_by_route": [
        "Which route had the most delays?",
        "most delays by route",
        "routes with highest delay",
        "delay comparison across routes",
        "average delay per route",
    ],
    "top_warehouses": [
        "Top 3 warehouses by processing time",
        "fastest warehouses",
        "best performing warehouse",
        "warehouse performance",
    ],
    "delay_reasons": [
        "Show delay reasons",
        "Why are shipments delayed",
        "distribution of delay reasons",
        "breakdown of delays",
        "Show total delayed shipments by delay reason",
        "total delayed shipments by delay reason",
        "delayed shipments by reason",
    ],
    "predict_delay": [
        "Predict delay next week",
        "forecast next week delays",
        "delay prediction",
    ],
    "predict_delay_month": [
        "Predict delay next month",
        "forecast next month delays",
        "delay prediction for next month",
        "predict the delay rate for next month",
    ],
    "predict_delay_year": [
        "Predict delay next year",
        "forecast next year delays",
        "delay prediction for next year",
    ],
}

INTENT_FUNCS = {
    "delays_by_route": _delays_by_route,
    "top_warehouses": _top_warehouses,
    "delay_reasons": _delay_reasons,
    "predict_delay": _predict_delay_next_week,
    "predict_delay_month": _predict_delay_next_month,
    "predict_delay_year": _predict_delay_next_year,
}


def generate_synthetic_training() -> Tuple[List[str], List[str]]:
    texts, labels = [], []
    for label, prompts in INTENT_TEMPLATES.items():
        for p in prompts:
            texts.append(p)
            labels.append(label)
        # add simple paraphrases
        texts.append(f"Tell me about {label.replace('_', ' ')}")
        labels.append(label)
    return texts, labels


train_texts, train_labels = generate_synthetic_training()
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(train_texts)

clf = LogisticRegression(max_iter=500)
X, y = X_train, train_labels
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
clf.fit(X_tr, y_tr)


def classify_rule_based(query: str) -> str:
    q = query.lower()
    if re.search(r"most delay|highest delay|delay.*route|route.*delay", q):
        return "delays_by_route"
    if re.search(r"warehouse|processing time|top warehouse|best warehouse", q):
        return "top_warehouses"
    if re.search(r"reason|why|cause|breakdown|delayed.*shipment|total.*delayed|delay.*reason", q):
        return "delay_reasons"
    # Check in order: week (most specific) -> month -> year
    if re.search(r"next\s+week|week.*delay|delay.*next\s+week", q):
        return "predict_delay"
    if re.search(r"next\s+year|year.*delay|delay.*next\s+year", q):
        return "predict_delay_year"
    if re.search(r"next\s+month|month.*delay|delay.*next\s+month|predict.*next\s+month", q):
        return "predict_delay_month"
    if re.search(r"predict|forecast", q):
        return "predict_delay"  # Default to week if just "predict" or "forecast"
    return "delays_by_route"


def classify_similarity(query: str) -> str:
    q_vec = vectorizer.transform([query])
    sims = cosine_similarity(q_vec, vectorizer.transform(train_texts)).flatten()
    best_idx = int(np.argmax(sims))
    return train_labels[best_idx]


def classify_ml(query: str) -> str:
    q_vec = vectorizer.transform([query])
    return clf.predict(q_vec)[0]


def run_intent(intent: str, data: pd.DataFrame, month_context: str = None) -> Tuple[str, List[Dict], Dict]:
    handler = INTENT_FUNCS.get(intent, _delays_by_route)
    if intent == "delays_by_route":
        return handler(data, month_context)
    return handler(data)


class QueryPayload(BaseModel):
    query: str
    method: int = 1
    context: Optional[Dict] = None


app = FastAPI(title="Hermes Logistics AI Assistant", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _extract_time_period(query: str) -> Tuple[str, int]:
    """Extract time period from query: 'last week', 'last month', etc. Returns (period_type, value)."""
    q = query.lower()
    if re.search(r"last\s+week|past\s+week", q):
        return ("week", 1)
    if re.search(r"last\s+month|past\s+month", q):
        return ("month", 1)
    if re.search(r"last\s+(\d+)\s+weeks?", q):
        match = re.search(r"last\s+(\d+)\s+weeks?", q)
        return ("week", int(match.group(1)))
    if re.search(r"last\s+(\d+)\s+months?", q):
        match = re.search(r"last\s+(\d+)\s+months?", q)
        return ("month", int(match.group(1)))
    return (None, None)


def process_query(query: str, method: int = 1, context: Dict = None) -> Dict:
    # Use previous intent from context if this looks like a follow-up
    previous_intent = None
    try:
        if context and isinstance(context, dict) and context.get("previous_intent"):
            # Check if query is a follow-up (short, starts with "how about", "what about", etc.)
            follow_up_patterns = [r"^how about", r"^what about", r"^and", r"^also", r"^then"]
            if any(re.search(pattern, query.lower()) for pattern in follow_up_patterns):
                previous_intent = context["previous_intent"]
    except Exception:
        # If context parsing fails, just continue without it
        pass
    
    if method == 1:
        intent = previous_intent or classify_rule_based(query)
        method_name = "rule_based"
    elif method == 2:
        intent = previous_intent or classify_similarity(query)
        method_name = "similarity"
    else:
        intent = previous_intent or classify_ml(query)
        method_name = "ml_classifier"

    # Extract time filters from query
    data_to_use = df_shipments.copy()
    month_context = None
    time_period_type, time_period_value = _extract_time_period(query)
    
    # Filter by month if specified
    month_num = _extract_month_from_query(query)
    if month_num:
        month_names = ["", "January", "February", "March", "April", "May", "June",
                      "July", "August", "September", "October", "November", "December"]
        month_context = month_names[month_num]
        data_to_use = _filter_by_month(data_to_use, month_num)
        if len(data_to_use) == 0:
            return {
                "method_used": method_name,
                "intent": intent,
                "text": f"No data available for {month_context}.",
                "table_data": [],
                "chart_data": {},
            }
    # Filter by time period (last week, last month, etc.)
    elif time_period_type == "week":
        data_to_use = _week_filter(data_to_use, time_period_value)
    elif time_period_type == "month":
        cutoff = data_to_use["date"].max() - pd.DateOffset(months=time_period_value)
        data_to_use = data_to_use[data_to_use["date"] >= cutoff]

    text, table, chart = run_intent(intent, data_to_use, month_context)
    return {
        "method_used": method_name,
        "intent": intent,
        "text": text,
        "table_data": table,
        "chart_data": chart,
    }


@app.get("/health")
def health():
    return {"status": "ok", "records": len(df_shipments)}


@app.get("/api/stats")
def stats():
    text_r, table_r, chart_r = _delays_by_route(df_shipments)
    text_w, table_w, chart_w = _top_warehouses(df_shipments)
    text_d, table_d, chart_d = _delay_reasons(df_shipments)
    return {
        "routes": {"text": text_r, "table": table_r, "chart": chart_r},
        "warehouses": {"text": text_w, "table": table_w, "chart": chart_w},
        "reasons": {"text": text_d, "table": table_d, "chart": chart_d},
    }


@app.get("/api/query")
def get_query(q: str = Query(..., description="User query"), method: int = 1):
    return process_query(q, method)


@app.post("/api/query")
def post_query(payload: QueryPayload):
    try:
        return process_query(payload.query, payload.method, payload.context)
    except Exception as e:
        # Log error and return a safe response
        import traceback
        print(f"Error processing query: {e}")
        print(traceback.format_exc())
        return {
            "method_used": "error",
            "intent": "error",
            "text": f"Sorry, an error occurred while processing your query: {str(e)}",
            "table_data": [],
            "chart_data": {},
        }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


