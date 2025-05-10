from util import get_player_ID, get_player_name, pitcher, get_nonpitcher_stats, get_pitcher_stats
import joblib
import os
import streamlit as st

@st.cache_resource
def predict_pitcher(stats):
    metrics = ["W", "L", "G", "GS", "CG", "SHO", "SV", "IPouts",
    "H", "ER", "HR", "BB", "SO", "BAOpp", "ERA", "IBB",
    "WP", "HBP", "BK", "BFP", "GF", "R", "SH", "SF",
    "GIDP", "WHIP"]

    X = stats[metrics]
    y = stats["HOF"]

    current_script_dir = os.path.dirname(__file__)
    pitching_scaler_relative_path = os.path.join(current_script_dir, "..", "Scalers", "pitching_scaler.pkl")
    pitching_model_relative_path = os.path.join(current_script_dir, "..", "Models", "pitching_model.pkl")
    scaler = joblib.load(pitching_scaler_relative_path)

    X = scaler.transform(X)
    model = joblib.load(pitching_model_relative_path)

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[0]
    print(predictions)
    print(probabilities)
    return predictions, probabilities

@st.cache_resource
def predict_nonpitcher(stats):
    
    metrics = ["AB", "R", "H", "2B", "3B", "HR", "RBI", "SB_x", "BB", "SO",
           "BA", "OBP", "SLG", "OPS", "G"]

    X = stats[metrics]
    y = stats["HOF"]

    current_script_dir = os.path.dirname(__file__)
    nonpitching_scaler_relative_path = os.path.join(current_script_dir, "..", "Scalers", "nonpitching_scaler.pkl")
    nonpitching_model_relative_path = os.path.join(current_script_dir, "..", "Models", "nonpitching_model.pkl")
    scaler = joblib.load(nonpitching_scaler_relative_path)

    X = scaler.transform(X)
    model = joblib.load(nonpitching_model_relative_path)

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[0]
    print(predictions)
    print(probabilities)
    return predictions, probabilities


def predict_player(player_name):
    player_id = get_player_ID(player_name)
    if not player_id:
        return "Invalid Player"

    is_pitcher = pitcher(player_id)

    if is_pitcher:
        stats = get_pitcher_stats(player_id)
        return predict_pitcher(stats)
    else:
        stats = get_nonpitcher_stats(player_id)
        return predict_nonpitcher(stats)
    

if __name__ == "__main__":
    print("Please input a player")
    player = input()
    predict_player(player)
        


