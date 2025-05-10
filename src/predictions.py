from util import get_player_ID, get_player_name, pitcher, get_nonpitcher_stats, get_pitcher_stats
import joblib


def predict_pitcher(stats):
    metrics = ["W", "L", "G", "GS", "CG", "SHO", "SV", "IPouts",
    "H", "ER", "HR", "BB", "SO", "BAOpp", "ERA", "IBB",
    "WP", "HBP", "BK", "BFP", "GF", "R", "SH", "SF",
    "GIDP", "WHIP"]

    X = stats[metrics]
    y = stats["HOF"]

    scaler = joblib.load("../Scalers/pitching_scaler.pkl")

    X = scaler.transform(X)
    model = joblib.load("../Models/pitching_model.pkl")

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[0]
    print(predictions)
    print(probabilities)
    return predictions, probabilities

def predict_nonpitcher(stats):
    
    metrics = ["AB", "R", "H", "2B", "3B", "HR", "RBI", "SB_x", "BB", "SO",
           "BA", "OBP", "SLG", "OPS", "G"]

    X = stats[metrics]
    y = stats["HOF"]

    scaler = joblib.load("../Scalers/nonpitching_scaler.pkl")

    X = scaler.transform(X)
    model = joblib.load("../Models/nonpitching_model.pkl")

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
        


