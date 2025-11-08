# fifa_app.py
"""
Dark-themed Streamlit UI for FIFA Match Predictor (Top 48 teams only)
with Tournament Simulation to predict overall World Cup Champion.
Run with:
    streamlit run fifa_app.py
"""

import streamlit as st
import pandas as pd
import joblib
import numpy as np
import random

# === Config / file paths ===
RANKINGS_PATH = r"C:/Users/Aaliya's 1st Laptop/Downloads/FIFA_predictor_datasets/fifa_ranking_2022-10-06.csv"
MODELS_DIR = "models"
SCALER_PATH = MODELS_DIR + "/scaler.pkl"
MODEL_RF_PATH = MODELS_DIR + "/model_rf.pkl"

# === Page setup ===
st.set_page_config(page_title="FIFA Match Predictor (Top 48)", layout="centered", page_icon="⚽", initial_sidebar_state="collapsed")

# === Dark theme CSS ===
dark_css = """
<style>
[data-testid="stAppViewContainer"] {
  background: linear-gradient(180deg, #0f1724 0%, #071021 100%);
  color: #e6eef8;
}
h1, h2, h3, h4, h5, h6, .stMarkdown { color: #e6eef8; }
[data-testid="stSidebar"] { background-color: #071021; }
.stButton>button {
  background: linear-gradient(90deg,#0ea5a4,#7c3aed);
  color: white;
  border-radius: 8px;
  padding: 8px 14px;
}
[data-testid="stMetricValue"] { color: #ffffff; font-weight: 700; }
small, .css-1v3fvcr { color: #b9c6d9; }
</style>
"""
st.markdown(dark_css, unsafe_allow_html=True)

# === Title/Header ===
st.markdown("<h1 style='text-align:center; margin-bottom:0.2rem;'>⚽ FIFA Match Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#bcd4ff;'>Predict match winners or simulate a full tournament among the Top 48 teams.</p>", unsafe_allow_html=True)
st.markdown("---", unsafe_allow_html=True)

# === Load rankings & prepare final 48 ===
@st.cache_data
def load_rankings(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    if 'team' not in df.columns:
        possible = [c for c in df.columns if 'country' in c.lower() or 'team' in c.lower()]
        if possible:
            df = df.rename(columns={possible[0]: 'team'})
        else:
            raise RuntimeError("Could not find 'team' column in rankings CSV.")
    for c in ['rank', 'points']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    df['team'] = df['team'].astype(str).str.strip()
    return df

rankings = load_rankings(RANKINGS_PATH)

qualified_28 = [
    'Canada', 'Mexico', 'USA', 'Australia', 'IR Iran', 'Japan', 'Jordan',
    'Korea Republic', 'Qatar', 'Saudi Arabia', 'Uzbekistan',
    'Algeria', 'Cabo Verde', "Côte d'Ivoire", 'Egypt', 'Ghana', 'Morocco',
    'Senegal', 'South Africa', 'Tunisia', 'Argentina', 'Brazil', 'Colombia',
    'Ecuador', 'Paraguay', 'Uruguay', 'New Zealand', 'England'
]

if 'rank' in rankings.columns:
    remaining = rankings[~rankings['team'].isin(qualified_28)].sort_values('rank').head(20)
else:
    remaining = rankings[~rankings['team'].isin(qualified_28)].sort_values('team').head(20)

final_48 = qualified_28 + remaining['team'].tolist()
available_final48 = [t for t in final_48 if t in rankings['team'].values]
teams = sorted(available_final48)

# === Load models ===
@st.cache_resource
def load_models(scaler_path, model_path):
    try:
        scaler = joblib.load(scaler_path)
    except Exception:
        scaler = None
    model = joblib.load(model_path)
    return scaler, model

try:
    scaler, model = load_models(SCALER_PATH, MODEL_RF_PATH)
except Exception as e:
    st.error(f"❌ Could not load model. Ensure models are saved in ./models\n\nError: {e}")
    st.stop()

# === Helper: build features ===
def build_features_hidden(home, away, rankings_df):
    hr = rankings_df[rankings_df['team'] == home]
    ar = rankings_df[rankings_df['team'] == away]
    if hr.empty or ar.empty:
        home_rank, away_rank, home_points, away_points = 100, 100, 1000, 1000
    else:
        home_rank = hr.iloc[0].get('rank', 100)
        away_rank = ar.iloc[0].get('rank', 100)
        home_points = hr.iloc[0].get('points', 1000)
        away_points = ar.iloc[0].get('points', 1000)
    rank_diff = home_rank - away_rank
    points_diff = home_points - away_points
    feat_vals = [rank_diff, points_diff, 1.4, 1.0, 40000, 0, 0]
    feat_cols = ['rank_diff','points_diff','home_xg','away_xg','Attendance','is_host','knockout']
    return pd.DataFrame([feat_vals], columns=feat_cols)

def predict_match(home, away):
    X_pred = build_features_hidden(home, away, rankings)
    try:
        probs = model.predict_proba(X_pred)[0]
        pred = model.predict(X_pred)[0]
    except Exception:
        if scaler is not None:
            Xs = scaler.transform(X_pred)
            probs = model.predict_proba(Xs)[0]
            pred = model.predict(Xs)[0]
        else:
            raise
    home_win_prob = float(probs[1])
    return home if home_win_prob > 0.5 else away

# === Manual match predictor ===
st.subheader("🎯 Single Match Prediction")
col1, col2 = st.columns(2)
with col1:
    home_team = st.selectbox("🏠 Home Team", teams, index=0)
with col2:
    away_team = st.selectbox("✈️ Away Team", teams, index=1 if len(teams)>1 else 0)

if st.button("🔮 Predict Winner (Top 48)"):
    if home_team == away_team:
        st.warning("Choose two different teams.")
    else:
        winner = predict_match(home_team, away_team)
        st.markdown(f"<div style='background:#0b1b2f; padding:18px; border-radius:10px;'>"
                    f"<h2 style='text-align:center;'>🏆 Predicted Winner: <span style='color:#98f0c7'>{winner}</span></h2>"
                    f"</div>", unsafe_allow_html=True)

# === Tournament Simulation ===
st.markdown("---")
st.subheader("🏆 Tournament Simulation — Predict World Cup Champion")

if st.button("🎲 Simulate Tournament"):
    with st.spinner("Simulating matches..."):
        teams_shuffled = random.sample(teams, len(teams))
        round_num = 1
        current_round = teams_shuffled
        results_log = []

        while len(current_round) > 1:
            round_winners = []
            round_pairs = []
            for i in range(0, len(current_round), 2):
                if i+1 >= len(current_round):
                    round_winners.append(current_round[i])
                    continue
                home, away = current_round[i], current_round[i+1]
                winner = predict_match(home, away)
                round_winners.append(winner)
                round_pairs.append((home, away, winner))
            results_log.append((round_num, round_pairs))
            round_num += 1
            current_round = round_winners

        champion = current_round[0]

    st.success(f"🏆 **Predicted FIFA World Cup Champion: {champion}**")
    st.markdown("### Tournament Results")
    for round_num, pairs in results_log:
        st.markdown(f"#### Round {round_num}")
        df_round = pd.DataFrame(pairs, columns=["Home", "Away", "Winner"])
        st.dataframe(df_round, use_container_width=True)

st.markdown("---")
st.caption("© 2025 FIFA Predictor | Random Forest Model")
