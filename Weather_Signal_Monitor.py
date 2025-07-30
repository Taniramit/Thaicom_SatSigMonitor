import streamlit as st
from streamlit_autorefresh import st_autorefresh
import pandas as pd
import numpy as np
import requests
import plotly.express as px
import plotly.graph_objects as go
import xgboost
import sklearn

import pickle
from datetime import datetime, timedelta
import os
import time

st.set_page_config(
    page_title="Signal Drop Monitor",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Custom CSS ---
st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-card {
        padding: 1.5rem;
        transition: transform 0.2s;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        height: 100%;
    }
    .prediction-card:hover { transform: translateY(-2px); }
    .danger-card {
        background: linear-gradient(135deg, #ff6b6b, #ee5a52);
        color: white;
        border: 2px solid #ff4757;
        height: 100%;
    }
    .safe-card {
        background: linear-gradient(135deg, #51cf66, #40c057);
        color: white;
        border: 2px solid #2ed573;
        height: 100%;
    }
    .metric-container {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #007bff;
    }
    .stTabs [data-baseweb="tab-list"] {
    }
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        padding: 0 2rem;
        background-color: #f1f3f4;
        color: #333;
        border-radius: 10px 10px 0 0;
    }
    .stTabs [aria-selected="true"] {
        background-color: #007bff;
        color: black;
    }
    </style>
""", unsafe_allow_html=True)


# -------------------- Main Features --------------------
MAIN_FEATURES = [
    "precipitation", "soil_moisture_9_to_27cm", "soil_temperature_54cm", 
    "dew_point_2m", "uv_index", "evapotranspiration", "wind_speed_80m", 
    "diffuse_radiation", "relative_humidity_2m", "shortwave_radiation"
]
# -------------------- Load ML Models --------------------
@st.cache_resource
def load_all_models():
    """Load pre-trained ML models along with their features and scalers."""
    model_files = {
        "Random Forest": "rf_model.pkl",
        "Linear SVC": "svc_model.pkl", 
        "XGBoost": "xgb_model.pkl"
    }

    models = {}
    for name, path in model_files.items():
        if not os.path.exists(path):
            st.warning(f"⚠️ {name} file not found: {path}")
            continue

        try:
            with open(path, "rb") as f:
                model_bundle = pickle.load(f)

            # Unpack model bundle
            if isinstance(model_bundle, dict):
                model = model_bundle.get("model")
                features = model_bundle.get("features", MAIN_FEATURES)
                scaler = model_bundle.get("scaler", None)
            else:
                model = model_bundle
                features = MAIN_FEATURES
                scaler = None

            models[name] = {
                "model": model,
                "features": features,
                "scaler": scaler
            }

        except Exception as e:
            st.error(f"❌ Error loading {name}: {e}")

    return models

# -------------------- Fetch Weather Forecast --------------------
@st.cache_data(ttl=3600)
def fetch_weather_forecast(latitude, longitude):
    """Fetch hourly weather forecast data for the next 24 hours."""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": ",".join([
            "precipitation", "soil_moisture_9_to_27cm", "soil_temperature_54cm", "dew_point_2m", "uv_index",
            "evapotranspiration", "wind_speed_80m", "diffuse_radiation", "relative_humidity_2m", "shortwave_radiation"
        ]),
        "timezone": "auto",
        "start_date": datetime.now().strftime("%Y-%m-%d"),
        "end_date": (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    }

    try:
        with st.spinner("🌤️ Fetching weather forecast data..."):
            response = requests.get(url, params=params, timeout=15)
            if response.status_code == 200:
                hourly = response.json().get("hourly", {})
                df = pd.DataFrame(hourly)
                if "time" in df.columns:
                    df["time"] = pd.to_datetime(df["time"])
                return df
            else:
                st.error(f"Failed to fetch weather data: HTTP {response.status_code}")
    except Exception as e:
        st.error(f"Error fetching weather data: {e}")

    return None
# -------------------- Prepare Input for ML Model --------------------
def prepare_model_input(weather_df, model_features):
    """Return DataFrame with expected feature order and filled missing columns."""
    if weather_df is None or model_features is None:
        return None

    df = weather_df.copy()

    # Add missing columns as 0.0
    for feature in model_features:
        if feature not in df.columns:
            st.warning(f"Feature '{feature}' not found. Using default value 0.")
            df[feature] = 0.0

    return df[model_features].fillna(0)

# -------------------- Predict with Model --------------------
def predict_with_model(model_data, input_df):
    """Make predictions using a model and return class + probability."""
    if model_data is None or input_df is None:
        return None, None

    try:
        model = model_data["model"]
        scaler = model_data.get("scaler")

        # Apply scaling if needed
        input_array = scaler.transform(input_df) if scaler else input_df

        # Predict class
        predictions = model.predict(input_array)

        # Predict probabilities
        if hasattr(model, 'predict_proba'):
            drop_prob = model.predict_proba(input_array)[:, 1]
        elif hasattr(model, 'decision_function'):
            scores = model.decision_function(input_array)
            drop_prob = 1 / (1 + np.exp(-scores))  # Sigmoid function
        else:
            drop_prob = np.array([np.nan] * len(predictions))

        return predictions, drop_prob

    except Exception as e:
        st.error(f"Error making predictions: {e}")
        return None, None

# -------------------- Monitoring History Update with Location --------------------
def update_monitoring_history(history, timestamp, predictions, probabilities, weather_conditions, location):
    """Append latest prediction to monitoring history (keep max 24 entries)."""
    new_entry = {
        "timestamp": timestamp,
        "location": location,
        "predictions": predictions.copy(),
        "probabilities": probabilities.copy(),
        "weather": weather_conditions.copy()
    }

    history.append(new_entry)
    return history[-24:]

# -------------------- Main Function --------------------

def main():

    st_autorefresh(interval=60 * 1000, key="datarefresh")

    # -------------------- Initialize Session State --------------------
    if 'monitoring_history' not in st.session_state:
        st.session_state['monitoring_history'] = []

    if 'last_update' not in st.session_state:
        st.session_state['last_update'] = datetime.now()

    if 'manual_results' not in st.session_state:
        st.session_state['manual_results'] = {}
    # -------------------- Header --------------------
    st.markdown("""
    <div class="main-header">
        <h1>📡 Signal Drop Monitor</h1>
        <p>Advanced ML-powered monitoring with real-time weather data</p>
    </div>
    """, unsafe_allow_html=True)

    # -------------------- Load Models --------------------
    models = load_all_models()
    if not models:
        st.error("❌ No models loaded. Please check your model files.")
        st.stop()
    # -------------------- Auto Refresh Logic (Every Hour) --------------------
    now = datetime.now()
    # -------------------- Sidebar Location Selection --------------------
    st.sidebar.header("📍 Location Settings")
    location = st.sidebar.radio("Select Location", ["Thaicom SJ Infinite Building", "Thaicom Lat Lum Kaeo Station"])

    if location == "Thaicom SJ Infinite Building":
        latitude, longitude = 13.809, 100.558
    else:
        latitude, longitude = 14.053, 100.331

    latitude = st.sidebar.number_input("Latitude", value=latitude, format="%.4f")
    longitude = st.sidebar.number_input("Longitude", value=longitude, format="%.4f")

    next_hour = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
    st.sidebar.info(f"⏰ Next prediction scheduled: {next_hour.strftime('%H:%M:%S')}")

    if st.sidebar.button("🔄 Refresh Now"):
        st.cache_data.clear()
        st.rerun()

    # -------------------- Tabs --------------------
    tab1, tab2 = st.tabs(["📊 Hourly Monitoring", "🔧 Manual Testing"])

    # -------------------- Tab 1: Hourly Monitoring --------------------
    with tab1:
        st.header("Real-time Signal Drop Monitoring")
        st.markdown("*Monitoring data every hour with weather forecast predictions*")

        weather_data = fetch_weather_forecast(latitude, longitude)
        if weather_data is None or weather_data.empty:
            st.error("❌ Unable to fetch weather forecast data. Please check your internet connection.")
            return

        # Find the row that matches the current hour (rounded down)
        current_time = datetime.now().replace(minute=0, second=0, microsecond=0)

        # Find index of the matching time in weather_data
        if "time" in weather_data.columns:
            match_row = weather_data[weather_data["time"] == current_time]
            if not match_row.empty:
                current_idx = match_row.index[0]
            else:
                st.warning("⚠️ No forecast data available for the current hour.")
                current_idx = 0
        else:
            st.warning("⚠️ Weather data missing 'time' column.")
            current_idx = 0

        all_predictions = {}
        all_probabilities = {}

        for model_name, model_data in models.items():
            model_input = prepare_model_input(weather_data, model_data["features"])
            if model_input is not None:
                predictions, probabilities = predict_with_model(model_data, model_input)
                if predictions is not None:
                    all_predictions[model_name] = predictions
                    all_probabilities[model_name] = probabilities

        if not all_predictions:
            st.error("❌ Unable to make predictions with current weather data.")
            return

        # -------------------- Current Conditions & Auto History Update --------------------
        current_weather = {
            'Precipitation': weather_data.get('precipitation', [0])[current_idx],
            'Humidity': weather_data.get('relative_humidity_2m', [0])[current_idx],
            'Wind Speed': weather_data.get('wind_speed_80m', [0])[current_idx],
            'UV Index': weather_data.get('uv_index', [0])[current_idx],
            'Evapotranspiration': weather_data.get('evapotranspiration', [0])[current_idx],
            'Diffuse Radiation': weather_data.get('diffuse_radiation', [0])[current_idx],
        }

        if not st.session_state.monitoring_history or (
            now.minute == 0 and (
                st.session_state['monitoring_history'][-1]['timestamp'].hour != now.hour or
                st.session_state['monitoring_history'][-1]['timestamp'].date() != now.date()
            )
        ):
            predictions = {m: all_predictions[m][current_idx] for m in models}
            probabilities = {m: all_probabilities[m][current_idx] for m in models}
            st.session_state.monitoring_history = update_monitoring_history(
                st.session_state.monitoring_history,
                now,
                predictions,
                probabilities,
                current_weather,
                location
            )

        # -------------------- Display Controls --------------------
        st.subheader("🎯 Choose Models to Display")
        selected_models = st.multiselect(
            "Select models:",
            options=list(models.keys()),
            default=list(models.keys())
        )

        if not selected_models:
            st.warning("Please select at least one model to display.")
            return

        # -------------------- Prediction Cards --------------------
        st.subheader("🚨 Signal Drop Predictions (Next Hour)")
        cols = st.columns(len(selected_models))
        for i, model_name in enumerate(selected_models):
            pred = all_predictions[model_name][current_idx]
            prob = all_probabilities[model_name][current_idx]
            color = "red" if pred == 1 else "green"
            status = "🚨 SIGNAL DROP" if pred == 1 else "✅ STABLE"
            bg = "rgba(255, 0, 0, 0.1)" if pred == 1 else "rgba(0, 255, 0, 0.1)"
            border = f"2px solid {color}"

            with cols[i]:
                st.markdown(f"""
                <div style="height: 25vh; display: flex; flex-direction: column; justify-content: center; 
                            align-items: center; border-radius: 10px; background-color: {bg}; 
                            border: {border}; text-align: center;">
                    <h3>{model_name}</h3>
                    <h2 style="color: {color}; margin: 0;">{status}</h2>
                    <h3 style="margin-top: 5px;">{prob:.1%} Risk</h3>
                </div>
                """, unsafe_allow_html=True)

        # -------------------- Weather Metrics --------------------
        st.subheader("🌤️ Current Weather Conditions")
        labels = list(current_weather.keys())
        values = list(current_weather.values())
        weather_cols = st.columns(len(labels))
        for col, label, value in zip(weather_cols, labels, values):
            unit = "%" if "Humidity" in label else "mm" if "Precipitation" in label else "m/s"
            col.metric(label, f"{value:.1f} {unit}")

        # -------------------- Risk History Chart --------------------
        if st.session_state.monitoring_history:
            st.subheader("📈 Signal Drop Risk History")
            fig = go.Figure()
            times = [entry['timestamp'] for entry in st.session_state.monitoring_history]
            for i, model_name in enumerate(selected_models):
                probs = [entry['probabilities'].get(model_name, 0) for entry in st.session_state.monitoring_history]
                fig.add_trace(go.Scatter(
                    x=times, y=probs, mode='lines+markers', name=model_name,
                    line=dict(width=3), marker=dict(size=8)
                ))
            fig.add_hline(y=0.5, line_dash="dot", line_color="red", annotation_text="Risk Threshold")
            fig.update_layout(
                title='Historical Signal Drop Risk Predictions',
                xaxis_title='Time', yaxis_title='Drop Probability',
                yaxis=dict(range=[0, 1]), height=500, hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)

        # -------------------- Monitoring Table --------------------
        if st.session_state.monitoring_history:
            st.subheader("📊 Monitoring History")
            history_data = []
            for entry in st.session_state.monitoring_history[-12:]:
                row = {
                    'Time': entry['timestamp'].strftime('%m/%d %H:%M'),
                    'Wind Speed': f"{entry['weather']['Wind Speed']:.1f} m/s",
                    'Humidity': f"{entry['weather']['Humidity']:.0f}%",
                    'Precipitation': f"{entry['weather']['Precipitation']:.1f} mm",
                    'UV Index': f"{entry['weather']['UV Index']:.1f}",
                    'Location': entry.get('location', 'Unknown')
                }
                for model_name in selected_models:
                    pred = entry['predictions'].get(model_name, 0)
                    prob = entry['probabilities'].get(model_name, 0)
                    row[f"{model_name}"] = f"{'🚨' if pred == 1 else '✅'} {prob:.1%}"
                history_data.append(row)
            st.dataframe(pd.DataFrame(history_data), use_container_width=True)

        # -------------------- Feature Importance --------------------
        st.subheader("🔍 Model Feature Importance")
        tabs = st.tabs(selected_models)
        for i, model_name in enumerate(selected_models):
            with tabs[i]:
                model_data = models[model_name]
                model = model_data["model"]
                features = model_data["features"]
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    df = pd.DataFrame({'Feature': features, 'Importance': importances})
                    df = df.sort_values("Importance", ascending=True)
                    fig = px.bar(df, x="Importance", y="Feature", orientation="h",
                                 title=f"{model_name} Feature Importance")
                    st.plotly_chart(fig, use_container_width=True)
                elif hasattr(model, 'coef_'):
                    df = pd.DataFrame({'Feature': features, 'Coefficient': abs(model.coef_[0])})
                    df = df.sort_values("Coefficient", ascending=True)
                    fig = px.bar(df, x="Coefficient", y="Feature", orientation="h",
                                 title=f"{model_name} Coefficients")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Feature importance not available for this model.")



#------------------------------ Tab 2: Manual Parameter Testing --------------------
    with tab2:
        st.header("🔧 Manual Parameter Testing")
        st.markdown("*Test specific environmental conditions - independent of monitoring tab*")
        st.info("💡 Manually input weather parameters to test signal drop predictions.")

        # Get feature list from first loaded model
        features = next(iter(models.values()))["features"]
        manual_input = {}

        # -------------------- Input Form --------------------
        with st.form("manual_input_form"):
            st.subheader("🌦️ Environmental Parameters")
            col1, col2 = st.columns(2)

            with col1:
                manual_input["precipitation"] = st.number_input("Precipitation (mm)", 0.0, 50.0, 0.0, step=0.1)
                manual_input["soil_moisture_9_to_27cm"] = st.number_input("Soil Moisture 9-27cm", 0.0, 1.0, 0.3, step=0.01)
                manual_input["soil_temperature_54cm"] = st.number_input("Soil Temperature 54cm (°C)", -10.0, 50.0, 15.0, step=0.1)
                manual_input["dew_point_2m"] = st.number_input("Dew Point (°C)", -20.0, 30.0, 10.0, step=0.1)
                manual_input["uv_index"] = st.number_input("UV Index", 0.0, 15.0, 3.0, step=0.1)

            with col2:
                manual_input["evapotranspiration"] = st.number_input("Evapotranspiration (mm)", 0.0, 10.0, 2.0, step=0.01)
                manual_input["wind_speed_80m"] = st.number_input("Wind Speed 80m (m/s)", 0.0, 50.0, 20.0, step=0.1)
                manual_input["diffuse_radiation"] = st.number_input("Diffuse Radiation (W/m²)", 0.0, 500.0, 100.0, step=0.5)
                manual_input["relative_humidity_2m"] = st.number_input("Relative Humidity (%)", 0.0, 100.0, 60.0, step=1.0)
                manual_input["shortwave_radiation"] = st.number_input("Shortwave Radiation (W/m²)", 0.0, 1000.0, 200.0, step=1.0)

            predict_button = st.form_submit_button("🔮 Predict Signal Status", use_container_width=True)

        # -------------------- Run Prediction --------------------
        if predict_button:
            input_df = pd.DataFrame([manual_input])
            manual_predictions = {}
            manual_probabilities = {}

            for model_name, model_data in models.items():
                model_input = input_df.copy()

                # Fill in missing features with 0.0 if needed
                for feature in model_data["features"]:
                    if feature not in model_input.columns:
                        model_input[feature] = 0.0
                model_input = model_input[model_data["features"]]

                # Predict
                predictions, probabilities = predict_with_model(model_data, model_input)
                if predictions is not None:
                    manual_predictions[model_name] = predictions[0]
                    manual_probabilities[model_name] = probabilities[0] if probabilities is not None else np.nan
                else:
                    st.error(f"❌ Error predicting with {model_name}.")

            # Save results in session
            st.session_state.manual_results = {
                "predictions": manual_predictions,
                "probabilities": manual_probabilities,
                "input": manual_input,
                "timestamp": datetime.now()
            }

        # -------------------- Display Results --------------------
        if st.session_state.manual_results:
            results = st.session_state.manual_results
            st.subheader("🎯 Prediction Results")

            result_cols = st.columns(len(results["predictions"]))
            for i, (model_name, prediction) in enumerate(results["predictions"].items()):
                probability = results["probabilities"][model_name]
                color = "red" if prediction == 1 else "green"
                label = "🚨 SIGNAL DROP" if prediction == 1 else "✅ STABLE"

                with result_cols[i]:
                    st.markdown(f"""
                    <div style="padding: 15px; border-radius: 10px; background-color: rgba({255 if prediction else 0}, {0 if prediction else 255}, 0, 0.1); border: 2px solid {color}; text-align: center;">
                        <h4>{model_name}</h4>
                        <h3 style="color: {color};">{label}</h3>
                        <h4>{probability:.1%} probability</h4>
                    </div>
                    """, unsafe_allow_html=True)

            # -------------------- Model Comparison --------------------
            if len(results["predictions"]) > 1:
                st.subheader("📊 Model Comparison")

                comparison_df = pd.DataFrame({
                    "Model": list(results["predictions"].keys()),
                    "Prediction": ["Signal Drop" if p == 1 else "Stable" for p in results["predictions"].values()],
                    "Probability": list(results["probabilities"].values())
                })

                fig = px.bar(
                    comparison_df, x="Model", y="Probability", color="Prediction",
                    color_discrete_map={"Signal Drop": "red", "Stable": "green"},
                    title="Signal Drop Probability by Model"
                )
                fig.add_hline(y=0.5, line_dash="dash", line_color="orange", annotation_text="Decision Threshold")
                st.plotly_chart(fig, use_container_width=True)

                # -------------------- Summary Metrics --------------------
                st.subheader("📈 Prediction Summary")
                col1, col2, col3 = st.columns(3)
                with col1:
                    avg_prob = np.mean(list(results["probabilities"].values()))
                    st.metric("Average Risk", f"{avg_prob:.1%}")
                with col2:
                    drop_count = sum(1 for p in results["predictions"].values() if p == 1)
                    st.metric("Models Predicting Drop", f"{drop_count}/{len(results['predictions'])}")
                with col3:
                    max_prob = max(results["probabilities"].values())
                    st.metric("Highest Risk", f"{max_prob:.1%}")

        # -------------------- Footer --------------------
        st.markdown("---")
        st.markdown(f"*Weather-Based Signal Monitor | Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Location: {latitude:.3f}, {longitude:.3f}*")


if __name__ == "__main__":
    main()
