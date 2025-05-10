import pandas as pd
import joblib


def pitcher(player):
    df = pd.read_csv("../Data/Fielding.csv")
    df = df[df["playerID"] == player]
    return df["POS"].values[0] == "P"

def get_player_ID(player):
    first, last = player.split(" ")
    df = pd.read_csv("../Data/Master.csv")
    df = df[(df["nameFirst"] == first) & (df["nameLast"] == last)]
    if df.empty:
        return None
    else:
        return df["playerID"].values[0]

def get_players():
    df = pd.read_csv("../Data/Master.csv")
    df["space"] = " "
    df["nameFull"] = df["nameFirst"] + df["space"] + df["nameLast"]
    df = df[df['nameFull'].str.count(' ') < 2]

    return df["nameFull"].sort_values(ascending=True)
    
def get_player_name(player_id):
    df = pd.read_csv("../Data/Master.csv")
    df = df[df["playerID"] == player_id]
    if df.empty:
        return None
    else:
        return f"{df['nameFirst'].values[0]} {df['nameLast'].values[0]}"

def get_pitcher_stats(player_id):
    df = pd.read_csv("../Data/MasterPitching.csv")
    df = df[df["playerID"] == player_id]
    return df

def get_nonpitcher_stats(player_id):
    df = pd.read_csv("../Data/MasterNonPitching.csv")
    df = df[df["playerID"] == player_id]
    return df

def get_feature_significance_df(pitcher):
    if pitcher:
        model = joblib.load("../Models/pitching_model.pkl")
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
        model = joblib.load("../Models/nonpitching_model.pkl")
        metrics = ["AB", "R", "H", "2B", "3B", "HR", "RBI", "SB_x", "BB", "SO",
           "BA", "OBP", "SLG", "OPS", "G"]
        feature_importance = pd.DataFrame({
            'Feature': metrics,
            'Importance': abs(model.coef_[0])
        })
        feature_importance = feature_importance.sort_values('Importance', ascending=False)
        return feature_importance

        