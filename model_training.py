# model_training.py
"""
Trains models from your existing pipeline and saves:
 - scaler.pkl
 - model_lr.pkl
 - model_rf.pkl

Adjust file paths at the top if needed.
"""
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# === Config / file paths ===
MATCHES_PATH = r"C:/Users/Aaliya's 1st Laptop/Downloads/FIFA_predictor_datasets/matches_1930_2022.csv"
RANKINGS_PATH = r"C:/Users/Aaliya's 1st Laptop/Downloads/FIFA_predictor_datasets/fifa_ranking_2022-10-06.csv"
TRANSFERMARKT_PATH = r"transfermarkt_team_stats_all-time.csv"  # adjust if needed

OUTPUT_DIR = "models"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Load data ===
matches = pd.read_csv(MATCHES_PATH)
matches.columns = matches.columns.str.strip()

# Filter to >= 1998 (attempt both columns Year or Date)
matches = matches[
    (matches['Year'] >= 1998) |
    (pd.to_datetime(matches.get('Date', pd.NA), errors='coerce').dt.year >= 1998)
].copy()

matches['Date'] = pd.to_datetime(matches.get('Date', pd.NaT), errors='coerce')

# Numeric conversions
num_cols = ['home_score', 'away_score', 'home_xg', 'away_xg', 'Attendance']
for col in num_cols:
    if col in matches.columns:
        matches[col] = pd.to_numeric(matches[col], errors='coerce')

matches = matches.drop_duplicates()
matches = matches.replace(r'^\s*$', pd.NA, regex=True)

# World cup & transfermarkt (kept in pipeline, but not required for training right now)
# world_cup = pd.read_csv(WORLD_CUP_PATH)  # optional
rankings = pd.read_csv(RANKINGS_PATH)
rankings.columns = rankings.columns.str.strip()
for col in ['rank', 'previous_rank', 'points', 'previous_points']:
    if col in rankings.columns:
        rankings[col] = pd.to_numeric(rankings[col], errors='coerce')
rankings = rankings.replace(r'^\s*$', pd.NA, regex=True)

# If 'team' column isn't present, try to detect it
if 'team' not in rankings.columns:
    possible = [c for c in rankings.columns if 'country' in c.lower() or 'team' in c.lower()]
    if possible:
        rankings = rankings.rename(columns={possible[0]: 'team'})
    else:
        raise ValueError("Could not find a team column in the rankings file. Please check column names.")

# Transfermarkt basic load (non-critical)
try:
    transfermarkt = pd.read_csv(TRANSFERMARKT_PATH)
    transfermarkt.columns = transfermarkt.columns.str.strip()
except Exception:
    transfermarkt = None

# === Label ===
if 'home_score' not in matches.columns or 'away_score' not in matches.columns:
    raise ValueError("Matches file missing home_score and/or away_score; cannot label outcomes.")

matches['result'] = (matches['home_score'] > matches['away_score']).astype(int)

# === Merge rankings on home and away ===
# First normalize rankings team names trimming whitespace
rankings['team'] = rankings['team'].astype(str).str.strip()

# We'll attempt to merge using the 'home_team' and 'away_team' columns
for col in ['home_team', 'away_team']:
    if col not in matches.columns:
        raise ValueError(f"Matches file missing required column: {col}")

# Merge - create copies of rankings with renamed columns
r_home = rankings[['team', 'rank', 'points']].copy().rename(columns={'team': 'home_team', 'rank': 'home_rank', 'points': 'home_points'})
r_away = rankings[['team', 'rank', 'points']].copy().rename(columns={'team': 'away_team', 'rank': 'away_rank', 'points': 'away_points'})

matches = matches.merge(r_home, on='home_team', how='left')
matches = matches.merge(r_away, on='away_team', how='left')

# === Derived features ===
matches['rank_diff'] = matches['home_rank'] - matches['away_rank']
matches['points_diff'] = matches['home_points'] - matches['away_points']

# is_host: some files have 'Host' column
matches['is_host'] = (matches.get('home_team') == matches.get('Host')).astype(int)

# knockout detection from Round column - be conservative if Round missing
if 'Round' in matches.columns:
    matches['knockout'] = matches['Round'].str.contains('Final|Semi|Quarter|Round of 16', case=False, na=False).astype(int)
else:
    matches['knockout'] = 0

features = ['rank_diff', 'points_diff', 'home_xg', 'away_xg', 'Attendance', 'is_host', 'knockout']

# Keep only rows where required features exist
model_data = matches.dropna(subset=features + ['result']).copy()
if model_data.shape[0] < 50:
    print("Warning: model_data has fewer than 50 rows after dropping NA — check data quality/path.")

X = model_data[features]
y = model_data['result']

# === Train / test split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Scale ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# === Models ===
# Logistic Regression
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train_scaled, y_train)

# Random Forest (uses unscaled features usually fine)
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)

# === Evaluate ===
print("=== Logistic Regression ===")
print(classification_report(y_test, log_reg.predict(X_test_scaled)))
print("=== Random Forest ===")
print(classification_report(y_test, rf.predict(X_test)))

print("Random Forest Confusion Matrix:")
print(confusion_matrix(y_test, rf.predict(X_test)))

# Feature importances for RF
try:
    importances = rf.feature_importances_
    for f, imp in zip(features, importances):
        print(f"{f}: {imp:.4f}")
except Exception:
    pass

# === Save models & scaler ===
joblib.dump(scaler, os.path.join(OUTPUT_DIR, "scaler.pkl"))
joblib.dump(log_reg, os.path.join(OUTPUT_DIR, "model_lr.pkl"))
joblib.dump(rf, os.path.join(OUTPUT_DIR, "model_rf.pkl"))

print(f"\nSaved scaler and models to '{OUTPUT_DIR}' folder.")
