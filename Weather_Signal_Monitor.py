import streamlit as st
from streamlit_autorefresh import st_autorefresh
import pandas as pd
import numpy as np
import requests
import plotly.express as px
import plotly.graph_objects as go
import pickle
from datetime import datetime, timedelta, timezone
import os
import math
from copy import deepcopy
import glob
import joblib

# Import the new component for tab management
from streamlit_option_menu import option_menu

# Import ML libraries
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
from imblearn.combine import SMOTETomek
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

# --- Page Configuration ---
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
        padding: 2rem; border-radius: 10px; color: white;
        text-align: center; margin-bottom: 2rem;
    }
    /* Style for the option_menu to look like standard tabs */
    div[data-testid="stOptionMenu"] > div > button {
        background-color: #f1f3f4;
        color: #333;
        border-radius: 10px 10px 0 0 !important;
    }
    div[data-testid="stOptionMenu"] > div > button[aria-selected="true"] {
        background-color: #007bff;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)


# =================================================================================
# SCRIPT CONFIGURATION (Single Source of Truth)
# =================================================================================

FINAL_FEATURES = [
    "precipitation (mm)", "lifted_index ()", "dew_point_2m (°C)", "cape (J/kg)",
    "soil_moisture_27_to_81cm (m³/m³)", "evapotranspiration (mm)",
    "soil_moisture_9_to_27cm (m³/m³)", "cloud_cover (%)", "surface_pressure (hPa)",
    "relative_humidity_2m (%)", "wind_speed_80m (m/s)", "temperature_2m (°C)"
]
API_FEATURES = [name.split(' ')[0] for name in FINAL_FEATURES]
MODEL_BASE_NAMES = {
    "Random Forest": "rf_model",
    "XGBoost": "xgb_model",
    "Linear SVC": "svc_model"
}
# Define Bangkok Timezone (UTC+7)
BKK_TZ = timezone(timedelta(hours=7))

# =================================================================================
# CORE FUNCTIONS
# =================================================================================

@st.cache_resource
def load_all_models():
    models = {}
    for name, base_filename in MODEL_BASE_NAMES.items():
        model_files = glob.glob(f"{base_filename}*.pkl")
        if not model_files:
            st.warning(f"⚠️ No model file for {name} found.")
            continue
        latest_file = sorted(model_files, reverse=True)[0]
        try:
            with open(latest_file, "rb") as f:
                bundle = joblib.load(f)
            bundle['features'] = FINAL_FEATURES
            models[name] = {**bundle, "filename": latest_file}
        except Exception as e:
            st.error(f"❌ Error loading {name} from `{latest_file}`: {e}")
    return models

@st.cache_data(ttl=3600)
def fetch_weather_data(latitude, longitude, start_date, end_date, api="forecast"):
    base_url = "https://api.open-meteo.com/v1/forecast" if api == "forecast" else "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": latitude, "longitude": longitude, "hourly": ",".join(API_FEATURES),
        "start_date": start_date.strftime("%Y-%m-%d"), "end_date": end_date.strftime("%Y-%m-%d"),
        "timezone": "Asia/Bangkok"
    }
    try:
        with st.spinner(f"🌤️ Fetching weather {api} data..."):
            response = requests.get(base_url, params=params, timeout=20)
            response.raise_for_status()
            df = pd.DataFrame(response.json()['hourly'])
            df["time"] = pd.to_datetime(df["time"])
            api_to_final_map = {api_feat: final_feat for api_feat, final_feat in zip(API_FEATURES, FINAL_FEATURES)}
            df.rename(columns=api_to_final_map, inplace=True)
            return df
    except Exception as e:
        st.error(f"Error fetching weather data: {e}")
        return pd.DataFrame()

def predict_with_model(model_data, input_df):
    if not isinstance(model_data, dict) or input_df is None or input_df.empty: return None, None
    try:
        model, scaler, threshold = model_data["model"], model_data["scaler"], model_data.get("threshold", 0.5)
        if not isinstance(input_df, pd.DataFrame):
            input_df = pd.DataFrame(input_df, columns=model_data["features"])
        input_df_ordered = input_df[model_data["features"]]
        input_scaled = scaler.transform(input_df_ordered)
        if hasattr(model, 'predict_proba'):
            scores = model.predict_proba(input_scaled)[:, 1]
        else:
            scores_raw = model.decision_function(input_scaled)
            scores = 1 / (1 + np.exp(-scores_raw))
        predictions = (scores >= threshold).astype(int)
        return predictions, scores
    except Exception as e:
        st.error(f"Prediction Error for {model_data.get('filename', 'model')}: {e}")
        return None, None

def update_monitoring_history(history, timestamp, predictions, probabilities, weather_conditions, location):
    new_entry = {
        "timestamp": timestamp, "location": location, "predictions": predictions.copy(),
        "probabilities": probabilities.copy(), "weather": weather_conditions.copy()
    }
    history.append(new_entry)
    return history[-24:]

# =================================================================================
# MAIN APP LAYOUT
# =================================================================================

def main():
    if 'monitoring_history' not in st.session_state: st.session_state['monitoring_history'] = []
    if 'last_location' not in st.session_state: st.session_state['last_location'] = None
    if 'retraining_results' not in st.session_state: st.session_state['retraining_results'] = None

    st.markdown("<div class='main-header'><h1>📡 Signal Drop Monitor</h1><p>Advanced ML-powered monitoring with real-time, location-aware weather data</p></div>", unsafe_allow_html=True)
    
    models = load_all_models()
    if not models: st.stop()

    with st.sidebar:
        st.header("📍 Location Settings")
        location_option = st.radio("Select Location", ["Thaicom Capital Tower", "Thaicom Lat Lum Kaeo Station"])
        lat_default, lon_default = (13.739, 100.547) if "Capital Tower" in location_option else (14.053, 100.331)
        latitude = st.number_input("Latitude", value=lat_default, format="%.4f")
        longitude = st.number_input("Longitude", value=lon_default, format="%.4f")
        
        now_bkk_sidebar = datetime.now(BKK_TZ)
        next_hour = (now_bkk_sidebar + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
        st.info(f"⏰ Next prediction scheduled: {next_hour.strftime('%H:%M:%S')}")
        
        if st.button("🔄 Refresh Data & Rerun"):
            st.cache_data.clear(); st.cache_resource.clear(); st.rerun()

    current_location_tuple = (location_option, latitude, longitude)
    if st.session_state.last_location != current_location_tuple:
        st.toast("Location changed. Fetching new data...")
        st.session_state.monitoring_history = []
        st.session_state.last_location = current_location_tuple
        st.cache_data.clear()

    selected_tab = option_menu(
        menu_title=None,
        options=["📊 Hourly Monitoring", "🔧 Manual Testing", "🦾 Model Retraining"],
        icons=['graph-up', 'wrench-adjustable-circle', 'robot'],
        menu_icon="cast", default_index=0, orientation="horizontal",
    )

    if selected_tab == "📊 Hourly Monitoring":
        st_autorefresh(interval=60 * 1000, key="datarefresh")
        st.header("Real-time Signal Drop Monitoring")
        weather_data = fetch_weather_data(latitude, longitude, datetime.now(BKK_TZ), datetime.now(BKK_TZ) + timedelta(days=1))
        if weather_data.empty: st.stop()

        now_bkk = datetime.now(BKK_TZ)
        current_hour_naive = now_bkk.replace(minute=0, second=0, microsecond=0, tzinfo=None)
        
        current_row = weather_data[weather_data["time"] == current_hour_naive]
        if current_row.empty:
            st.warning("Data for current hour not available. Using first forecast hour.")
            current_row = weather_data.iloc[[0]]

        all_predictions, all_probabilities = {}, {}
        for name, data in models.items():
            preds, probs = predict_with_model(data, current_row)
            if preds is not None:
                all_predictions[name] = preds[0]; all_probabilities[name] = probs[0]
        
        is_preset_location = math.isclose(latitude, lat_default) and math.isclose(longitude, lon_default)
        location_display_name = location_option if is_preset_location else f"{latitude:.3f}, {longitude:.3f}"

        if not st.session_state.monitoring_history or (st.session_state.monitoring_history[-1]['timestamp'].hour != now_bkk.hour):
            current_weather_values = current_row[FINAL_FEATURES].iloc[0].to_dict()
            st.session_state.monitoring_history = update_monitoring_history(
                st.session_state.monitoring_history, now_bkk, all_predictions, all_probabilities, current_weather_values, location_display_name
            )
        
        st.subheader("🌤️ Current Weather Conditions")
        cols = st.columns(4)
        for i, feature in enumerate(FINAL_FEATURES):
            cols[i % 4].metric(label=feature, value=f"{current_row.iloc[0].get(feature, 0):.2f}")
        
        st.divider()
        st.subheader("🎯 Choose Models to Display")
        selected_models = st.multiselect("Select models:", options=list(models.keys()), default=list(models.keys()))

        if selected_models:
            st.subheader("🚨 Signal Drop Predictions (Current Hour)")
            pred_cols = st.columns(len(selected_models))
            for i, name in enumerate(selected_models):
                pred, prob = all_predictions.get(name, 0), all_probabilities.get(name, 0)
                status, color, bg = ("🚨 SIGNAL DROP", "red", "rgba(255,0,0,0.1)") if pred == 1 else ("✅ STABLE", "green", "rgba(0,255,0,0.1)")
                with pred_cols[i]:
                    st.markdown(f'<div style="border: 2px solid {color}; background-color: {bg}; border-radius: 10px; padding: 1rem; text-align: center; height: 100%;"><h3>{name}</h3><h2 style="color:{color};">{status}</h2><h4>{prob:.1%} probability</h4></div>', unsafe_allow_html=True)
        
        st.divider()
        if st.session_state.monitoring_history and selected_models:
            st.subheader("📊 Hourly Prediction History")
            history_data = []
            for entry in st.session_state.monitoring_history:
                row = {'Time': entry['timestamp'].strftime('%m/%d/%y %H:%M'), 'Location': entry['location']}
                prob_values = [entry['probabilities'].get(name, 0) for name in selected_models if name in entry['probabilities']]
                for name in selected_models:
                    if name in entry['predictions']:
                        row[name] = f"{'🚨' if entry['predictions'][name] else '✅'} ({entry['probabilities'][name]:.0%})"
                if prob_values:
                    avg_prob = sum(prob_values) / len(prob_values)
                    row["Average"] = f"{'🚨' if avg_prob > 0.5 else '✅'} ({avg_prob:.0%})"
                history_data.append(row)
            st.dataframe(pd.DataFrame(history_data), use_container_width=True)

            chart_df = pd.DataFrame([{"time": e['timestamp'], "model": name, "probability": prob} for e in st.session_state.monitoring_history for name, prob in e['probabilities'].items() if name in selected_models])
            if not chart_df.empty:
                fig = px.line(chart_df, x='time', y='probability', color='model', markers=True, title="Historical Signal Drop Risk")
                fig.add_hline(y=0.5, line_dash="dot", line_color="red", annotation_text="Risk Threshold")
                st.plotly_chart(fig, use_container_width=True)

            st.divider()
            st.subheader("🔍 Model Feature Importance")
            fi_tabs = st.tabs(selected_models)
            for i, name in enumerate(selected_models):
                with fi_tabs[i]:
                    model_obj, features = models[name]["model"], models[name]["features"]
                    if hasattr(model_obj, 'feature_importances_'):
                        df = pd.DataFrame({'Feature': features, 'Importance': model_obj.feature_importances_}).sort_values("Importance", ascending=True)
                        st.plotly_chart(px.bar(df, x="Importance", y="Feature", orientation="h", title=f"{name} Feature Importance"), use_container_width=True)
                    elif hasattr(model_obj, 'coef_'):
                        df = pd.DataFrame({'Feature': features, 'Coefficient': abs(model_obj.coef_[0])}).sort_values("Coefficient", ascending=True)
                        st.plotly_chart(px.bar(df, x="Coefficient", y="Feature", orientation="h", title=f"{name} Feature Coefficients"), use_container_width=True)

    if selected_tab == "🔧 Manual Testing":
        st.header("🔧 Manual Parameter Testing")
        with st.form("manual_input_form"):
            manual_input = {}
            cols = st.columns(4)
            for i, feature in enumerate(FINAL_FEATURES):
                manual_input[feature] = cols[i % 4].number_input(feature, value=0.0, step=0.1, key=f"manual_{feature}")
            if st.form_submit_button("🔮 Predict Signal Status", use_container_width=True, type="primary"):
                input_df = pd.DataFrame([manual_input])
                st.subheader("🎯 Manual Prediction Results")
                res_cols = st.columns(len(models))
                for i, (name, data) in enumerate(models.items()):
                    pred, prob = predict_with_model(data, input_df)
                    with res_cols[i]:
                        if pred is not None:
                            status, color, bg = ("🚨 SIGNAL DROP", "red", "rgba(255,0,0,0.1)") if pred[0] == 1 else ("✅ STABLE", "green", "rgba(0,255,0,0.1)")
                            st.markdown(f'<div style="border: 2px solid {color}; background-color: {bg}; border-radius: 10px; padding: 1rem; text-align: center; height: 100%;"><h3>{name}</h3><h2 style="color:{color};">{status}</h2><h4>{prob[0]:.1%} probability</h4></div>', unsafe_allow_html=True)
                            
    if selected_tab == "🦾 Model Retraining":
        st.header("🦾 Model Retraining with New Data")
        
        uploaded_file = st.file_uploader("📁 Upload CSV ('time' or '_time', and 'Rx_EsNo')", type=["csv"])
        c1, c2 = st.columns(2)
        user_lat = c1.number_input("Latitude of data location", format="%.4f", value=15.4719, key="retrain_lat")
        user_lon = c2.number_input("Longitude of data location", format="%.4f", value=98.6433, key="retrain_lon")
        
        st.divider()
        
        st.subheader("1. Select Base Model Versions for Retraining")
        selected_model_files = {}
        for name, base_name in MODEL_BASE_NAMES.items():
            versions = sorted(glob.glob(f"{base_name}*.pkl"))
            if not versions:
                st.warning(f"No versions found for {name}.")
                selected_model_files[name] = None
            else:
                selected_model_files[name] = st.selectbox(f"Choose base model for **{name}**:", options=versions, index=len(versions) - 1, format_func=os.path.basename)
        
        if st.button("🚀 Start Retraining", disabled=(uploaded_file is None), use_container_width=True, type="primary"):
            def haversine_km(lat1, lon1, lat2, lon2):
                lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
                a = math.sin((lat2-lat1)/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin((lon2-lon1)/2)**2
                return 6371 * 2 * math.asin(math.sqrt(a))
            
            with st.spinner("Executing new retraining pipeline..."):
                STATION_LAT, STATION_LON, DROP_DELTA_DB, RANDOM_STATE = 14.0535, 100.332, 1.3, 42
                
                signal_df = pd.read_csv(uploaded_file)
                if "_time" in signal_df.columns: signal_df.rename(columns={"_time": "time"}, inplace=True)
                if "Rx EsNo" in signal_df.columns: signal_df.rename(columns={"Rx EsNo": "Rx_EsNo"}, inplace=True)
                
                signal_df["time"] = pd.to_datetime(signal_df["time"], errors='coerce').dt.floor('h')
                signal_df.dropna(subset=["time", "Rx_EsNo"], inplace=True)
                
                distance = haversine_km(user_lat, user_lon, STATION_LAT, STATION_LON)
                baseline_db = -0.0087 * distance + 17.77
                signal_df["drop"] = (signal_df["Rx_EsNo"] <= (baseline_db - DROP_DELTA_DB)).astype(int)

                start_date, end_date = signal_df['time'].min(), signal_df['time'].max()
                api_to_use = "archive" if start_date < (datetime.now(BKK_TZ) - timedelta(days=60)) else "forecast"
                st.info(f"Data is {'older than 60 days' if api_to_use == 'archive' else 'recent'}. Using the Open-Meteo **{api_to_use}** API.")

                weather_df = fetch_weather_data(user_lat, user_lon, start_date, end_date, api=api_to_use)
                if weather_df.empty: st.stop()

                missing_features = set(FINAL_FEATURES) - set(weather_df.columns)
                if missing_features:
                    st.warning(f"API did not return: {', '.join(missing_features)}. These will be filled with 0.")
                    for feature in missing_features: weather_df[feature] = 0.0

                merged_df = pd.merge(signal_df, weather_df, on="time", how="inner")
                X_all = merged_df[FINAL_FEATURES]; y_all = merged_df["drop"]

                drop_counts = y_all.value_counts()
                st.info(f"Data Labeling Results: Stable Signals (0): {drop_counts.get(0, 0)}, Signal Drops (1): {drop_counts.get(1, 0)}")
                if y_all.nunique() < 2:
                    st.error("Training Failed: The provided data resulted in only one class."); st.stop()
                
                X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=RANDOM_STATE, stratify=y_all)
                
                imputer = SimpleImputer(strategy='median')
                X_train_imputed = imputer.fit_transform(X_train)
                X_test_imputed = imputer.transform(X_test)
                
                retrained_models, performance_data = {}, []

                for name, filepath in selected_model_files.items():
                    if filepath is None: continue
                    with open(filepath, "rb") as f:
                        original_bundle = joblib.load(f)
                    original_bundle['features'] = FINAL_FEATURES
                    
                    y_pred_orig, _ = predict_with_model(original_bundle, pd.DataFrame(X_test, columns=FINAL_FEATURES))
                    
                    new_scaler = StandardScaler().fit(X_train_imputed)
                    X_train_scaled, X_test_scaled = new_scaler.transform(X_train_imputed), new_scaler.transform(X_test_imputed)
                    
                    if y_train.value_counts().min() > 6:
                        smt = SMOTETomek(random_state=RANDOM_STATE)
                        X_train_bal, y_train_bal = smt.fit_resample(X_train_scaled, y_train)
                    else:
                        X_train_bal, y_train_bal = X_train_scaled, y_train

                    new_model = deepcopy(original_bundle['model'])
                    new_model.fit(X_train_bal, y_train_bal)
                    y_pred_new = new_model.predict(X_test_scaled)

                    performance_data.append({
                        "Model": name, "Original F1": f1_score(y_test, y_pred_orig), "Retrained F1": f1_score(y_test, y_pred_new),
                        "Original Precision": precision_score(y_test, y_pred_orig, zero_division=0), "Retrained Precision": precision_score(y_test, y_pred_new, zero_division=0),
                        "Original Recall": recall_score(y_test, y_pred_orig, zero_division=0), "Retrained Recall": recall_score(y_test, y_pred_new, zero_division=0)
                    })
                    retrained_models[name] = {"features": FINAL_FEATURES, "scaler": new_scaler, "model": new_model, "threshold": 0.5}
                
                st.session_state.retraining_results = {"performance": performance_data, "models": retrained_models}
                st.rerun()

        if st.session_state.retraining_results:
            st.subheader("2. Performance Comparison & Model Selection")
            with st.form("model_selection_form"):
                user_choices = {}
                perf_df = pd.DataFrame(st.session_state.retraining_results["performance"])

                for _, row in perf_df.iterrows():
                    name = row["Model"]
                    st.markdown(f"--- \n ### 🔍 **{name}**")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Original Model Performance**")
                        st.metric("F1-Score", f"{row['Original F1']:.3f}")
                    with col2:
                        st.markdown("**Retrained Model Performance**")
                        st.metric("F1-Score", f"{row['Retrained F1']:.3f}", delta=f"{row['Retrained F1'] - row['Original F1']:.3f}")
                    
                    user_choices[name] = st.radio(f"Choose version for **{name}**:", ("Keep Original", "Save Retrained"), key=f"select_{name}", horizontal=True)
                
                st.divider()
                if st.form_submit_button("💾 Save All Chosen Versions", use_container_width=True, type="primary"):
                    with st.spinner("Saving models..."):
                        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                        for name, choice in user_choices.items():
                            if choice == "Save Retrained":
                                new_filename = f"{MODEL_BASE_NAMES[name]}_{timestamp}.pkl"
                                full_path = os.path.abspath(new_filename)
                                joblib.dump(st.session_state.retraining_results["models"][name], new_filename)
                                st.success(f"✅ Saved **{name}** to `{full_path}`.")
                            else:
                                st.info(f"Kept original model for **{name}**.")
                    
                    del st.session_state.retraining_results
                    st.success("Process complete! Refresh the page to load the latest models.")
                    st.balloons()

if __name__ == "__main__":
    main()