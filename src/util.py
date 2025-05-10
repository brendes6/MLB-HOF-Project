import pandas as pd
import joblib
import os


def pitcher(player):
    current_script_dir = os.path.dirname(__file__)
    fielding_relative_path = os.path.join(current_script_dir, "..", "Data", "Fielding.csv")
    df = pd.read_csv(fielding_relative_path)
    df = df[df["playerID"] == player]
    return df["POS"].values[0] == "P"

def get_player_ID(player):
    current_script_dir = os.path.dirname(__file__)
    master_relative_path = os.path.join(current_script_dir, "..", "Data", "Master.csv")
    first, last = player.split(" ")
    df = pd.read_csv(master_relative_path)
    df = df[(df["nameFirst"] == first) & (df["nameLast"] == last)]
    if df.empty:
        return None
    else:
        return df["playerID"].values[0]

def get_players():
    current_script_dir = os.path.dirname(__file__)
    master_relative_path = os.path.join(current_script_dir, "..", "Data", "Master.csv")
    df = pd.read_csv(master_relative_path)
    df["space"] = " "
    df["nameFull"] = df["nameFirst"] + df["space"] + df["nameLast"]
    df = df[df['nameFull'].str.count(' ') < 2]

    return df["nameFull"].sort_values(ascending=True)
    
def get_player_name(player_id):
    current_script_dir = os.path.dirname(__file__)
    master_relative_path = os.path.join(current_script_dir, "..", "Data", "Master.csv")
    df = pd.read_csv(master_relative_path)
    df = df[df["playerID"] == player_id]
    if df.empty:
        return None
    else:
        return f"{df['nameFirst'].values[0]} {df['nameLast'].values[0]}"

def get_pitcher_stats(player_id):
    current_script_dir = os.path.dirname(__file__)
    master_pitching_relative_path = os.path.join(current_script_dir, "..", "Data", "MasterPitching.csv")
    df = pd.read_csv(master_pitching_relative_path)
    df = df[df["playerID"] == player_id]
    return df

def get_nonpitcher_stats(player_id):
    current_script_dir = os.path.dirname(__file__)
    master_nonpitching_relative_path = os.path.join(current_script_dir, "..", "Data", "MasterNonPitching.csv")
    df = pd.read_csv(master_nonpitching_relative_path)
    df = df[df["playerID"] == player_id]
    return df

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
        feature_importance = pd.DataFrame({
            'Feature': metrics,
            'Importance': abs(model.coef_[0])
        })
        feature_importance = feature_importance.sort_values('Importance', ascending=False)
        return feature_importance
    else:
        model = joblib.load(nonpitching_model_relative_path)
        metrics = ["AB", "R", "H", "2B", "3B", "HR", "RBI", "SB_x", "BB", "SO",
           "BA", "OBP", "SLG", "OPS", "G"]
        feature_importance = pd.DataFrame({
            'Feature': metrics,
            'Importance': abs(model.coef_[0])
        })
        feature_importance = feature_importance.sort_values('Importance', ascending=False)
        return feature_importance

        