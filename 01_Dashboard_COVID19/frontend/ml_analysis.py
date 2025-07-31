# ml_analysis.py (modifié)
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def get_country_data(session, country):
    rows = session.execute("""
        SELECT date, total_cases FROM stats WHERE country=%s
    """, (country,))
    df = pd.DataFrame(rows.all())
    if df.empty:
        raise ValueError(f"Aucune donnée pour le pays {country}")
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    return df

def prepare_features(df):
    df = df.copy()
    df['day_number'] = (df['date'] - df['date'].min()).dt.days
    X = df['day_number'].values.reshape(-1, 1)
    y = df['total_cases'].values
    return X, y

def detect_trend_changes(y, window=7):
    if len(y) < window * 2:
        return []
    rolling_diff = pd.Series(y).rolling(window=window).mean().diff()
    threshold = rolling_diff.std() * 2
    change_points = []
    for i in range(window, len(rolling_diff) - window):
        if abs(rolling_diff.iloc[i]) > threshold:
            change_points.append(i)
    return change_points

def calculate_confidence_interval(predictions, mae, confidence=0.95):
    from scipy import stats
    alpha = 1 - confidence
    t_value = stats.t.ppf(1 - alpha/2, df=len(predictions)-1)
    margin_error = t_value * mae
    lower_bound = predictions - margin_error
    upper_bound = predictions + margin_error
    return lower_bound, upper_bound

def select_best_model(X, y, test_size=0.2):
    if len(X) < 20:
        return LinearRegression(), "linear"
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
    models = {
        'linear': LinearRegression(),
        'polynomial_2': Pipeline([('poly', PolynomialFeatures(degree=2)), ('linear', LinearRegression())]),
        'polynomial_3': Pipeline([('poly', PolynomialFeatures(degree=3)), ('linear', LinearRegression())])
    }
    best_score = -np.inf
    best_model = None
    best_name = "linear"
    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            score = r2_score(y_test, y_pred)
            if name.startswith('polynomial') and score < 0.7:
                score *= 0.8
            if score > best_score:
                best_score = score
                best_model = model
                best_name = name
        except:
            continue
    return best_model if best_model else LinearRegression(), best_name

def predict_cases(session, country, days_ahead=7):
    df = get_country_data(session, country)
    if len(df) < 10:
        raise ValueError("Pas assez de données pour le pays sélectionné")
    X, y = prepare_features(df)
    model = LinearRegression()
    model.fit(X, y)
    last_day = X[-1][0]
    future_days = np.array([last_day + i for i in range(1, days_ahead + 1)]).reshape(-1, 1)
    predictions = model.predict(future_days)
    predictions = np.clip(predictions, 0, None)
    return future_days.flatten(), predictions

def enhanced_predict_cases_df(df, days_ahead=7, model_type="polynomial"):
    if len(df) < 10:
        raise ValueError("Pas assez de données dans le DataFrame fourni (minimum 10 points)")
    X, y = prepare_features(df)
    if len(np.unique(y)) == 1:
        predictions = np.full(days_ahead, y[0])
        last_day = X[-1][0]
        future_days = np.array([last_day + i for i in range(1, days_ahead + 1)])
        confidence_lower = predictions * 0.95
        confidence_upper = predictions * 1.05
        return future_days, predictions, 0.0, confidence_lower, confidence_upper, 1.0, "constant"

    if model_type == "auto":
        model, model_name = select_best_model(X, y)
    elif model_type == "polynomial":
        degree = min(3, len(X) // 5)
        degree = max(2, degree)
        model = Pipeline([('poly', PolynomialFeatures(degree=degree)), ('linear', LinearRegression())])
        model_name = f"polynomial_{degree}"
    else:
        model = LinearRegression()
        model_name = "linear"

    try:
        model.fit(X, y)
    except:
        model = LinearRegression()
        model.fit(X, y)
        model_name = "linear_fallback"

    y_pred_train = model.predict(X)
    mae = mean_absolute_error(y, y_pred_train)
    r2 = r2_score(y, y_pred_train)
    last_day = X[-1][0]
    future_days = np.array([last_day + i for i in range(1, days_ahead + 1)]).reshape(-1, 1)

    try:
        predictions = model.predict(future_days)
    except:
        model = LinearRegression()
        model.fit(X, y)
        predictions = model.predict(future_days)
        model_name = "linear_fallback"
        mae = mean_absolute_error(y, model.predict(X))
        r2 = r2_score(y, model.predict(X))

    predictions = np.clip(predictions, 0, None)
    confidence_lower, confidence_upper = calculate_confidence_interval(predictions, mae)
    confidence_lower = np.clip(confidence_lower, 0, None)

    if len(y) >= 14:
        recent_trend = np.polyfit(range(len(y)-7, len(y)), y[-7:], 1)[0]
        overall_trend = np.polyfit(range(len(y)), y, 1)[0]
        if abs(recent_trend - overall_trend) > abs(overall_trend) * 0.5:
            trend_adjustment = recent_trend - overall_trend
            trend_effect = np.array([trend_adjustment * i for i in range(1, days_ahead + 1)])
            predictions += trend_effect
            predictions = np.clip(predictions, 0, None)

    # ✅ Retour de 7 éléments maintenant
    return (
        future_days.flatten(),
        predictions,
        mae,
        confidence_lower,
        confidence_upper,
        max(0, r2),
        model_name
    )

def predict_cases_df(df, days_ahead=7):
    if len(df) < 10:
        raise ValueError("Pas assez de données dans le DataFrame fourni")
    X, y = prepare_features(df)
    model = LinearRegression()
    model.fit(X, y)
    last_day = X[-1][0]
    future_days = np.array([last_day + i for i in range(1, days_ahead + 1)]).reshape(-1, 1)
    predictions = model.predict(future_days)
    predictions = np.clip(predictions, 0, None)
    return future_days.flatten(), predictions

def analyze_data_quality(df):
    quality_report = {
        'total_points': len(df),
        'date_range': (df['date'].min(), df['date'].max()),
        'missing_values': df.isnull().sum().sum(),
        'zero_values': (df['total_cases'] == 0).sum(),
        'negative_values': (df['total_cases'] < 0).sum(),
        'data_consistency': True
    }
    if not df['total_cases'].is_monotonic_increasing:
        decreases = df['total_cases'].diff() < -1000
        if decreases.sum() > 0:
            quality_report['data_consistency'] = False
            quality_report['significant_decreases'] = decreases.sum()
    return quality_report
