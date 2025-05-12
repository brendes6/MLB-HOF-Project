import pandas as pd
import joblib
import os
import streamlit as st

# Load specific data anc cache it on site
@st.cache_data
def load_master_data():
    current_script_dir = os.path.dirname(__file__)
    master_relative_path = os.path.join(current_script_dir, "..", "Data", "Master.csv")
    try:
        return pd.read_csv(master_relative_path)
    except FileNotFoundError:
        st.error("Master data file not found. Please ensure the data files are in the correct location.")
        return None

@st.cache_data
def load_fielding_data():
    current_script_dir = os.path.dirname(__file__)
    fielding_relative_path = os.path.join(current_script_dir, "..", "Data", "Fielding.csv")
    try:
        return pd.read_csv(fielding_relative_path)
    except FileNotFoundError:
        st.error("Fielding data file not found. Please ensure the data files are in the correct location.")
        return None

@st.cache_data
def load_master_pitching_data():
    current_script_dir = os.path.dirname(__file__)
    master_pitching_relative_path = os.path.join(current_script_dir, "..", "Data", "MasterPitching.csv")
    try:
        return pd.read_csv(master_pitching_relative_path)
    except FileNotFoundError:
        st.error("Master pitching data file not found. Please ensure the data files are in the correct location.")
        return None

@st.cache_data
def load_master_nonpitching_data():
    current_script_dir = os.path.dirname(__file__)
    master_nonpitching_relative_path = os.path.join(current_script_dir, "..", "Data", "MasterNonPitching.csv")
    try:
        return pd.read_csv(master_nonpitching_relative_path)
    except FileNotFoundError:
        st.error("Master non-pitching data file not found. Please ensure the data files are in the correct location.")
        return None

# helper functions for getting specific data
def pitcher(player):
    df = load_fielding_data()
    if df is None:
        return False
    df = df[df["playerID"] == player]
    if df.empty:
        return False
    return df["POS"].values[0] == "P"

def get_player_ID(player):
    df = load_master_data()
    if df is None:
        return None
    first, last = player.split(" ")
    df = df[(df["nameFirst"] == first) & (df["nameLast"] == last)]
    if df.empty:
        return None
    else:
        return df["playerID"].values[0]

def get_players():
    df = load_master_data()
    if df is None:
        return pd.Series([])
    df["space"] = " "
    df["nameFull"] = df["nameFirst"] + df["space"] + df["nameLast"]
    df = df[df['nameFull'].str.count(' ') < 2]
    return df["nameFull"].sort_values(ascending=True)
    
def get_player_name(player_id):
    df = load_master_data()
    if df is None:
        return None
    df = df[df["playerID"] == player_id]
    if df.empty:
        return None
    else:
        return f"{df['nameFirst'].values[0]} {df['nameLast'].values[0]}"

def get_pitcher_stats(player_id):
    df = load_master_pitching_data()
    if df is None:
        return pd.DataFrame()
    df = df[df["playerID"] == player_id]
    return df

def get_nonpitcher_stats(player_id):
    df = load_master_nonpitching_data()
    if df is None:
        return pd.DataFrame()
    df = df[df["playerID"] == player_id]
    return df

# Get feature signifiance dataframe quickly - used in app
@st.cache_resource
def get_feature_significance_df(pitcher):
    current_script_dir = os.path.dirname(__file__)
    pitching_model_relative_path = os.path.join(current_script_dir, "..", "Models", "pitching_model.pkl")
    nonpitching_model_relative_path = os.path.join(current_script_dir, "..", "Models", "nonpitching_model.pkl")

    if pitcher:
        model = joblib.load(pitching_model_relative_path)
        metrics = ["W", "L", "G", "GS", "CG", "SHO", "SV", "IPouts",
        "H", "ER", "HR", "BB", "SO", "BAOpp", "ERA", "IBB",
        "WP", "HBP", "BK", "BFP", "GF", "R", "SH", "SF",
        "GIDP", "WHIP"]
    else:
        model = joblib.load(nonpitching_model_relative_path)
        metrics = ["AB", "R", "H", "2B", "3B", "HR", "RBI", "SB_x", "BB", "SO",
            "BA", "OBP", "SLG", "OPS", "G"]

    feature_importance = pd.DataFrame({
        'Feature': metrics,
        'Importance': abs(model.coef_[0])
    })
    return feature_importance.sort_values('Importance', ascending=False)

        