import pandas as pd
import numpy as np


batting_data = pd.read_csv("../Data/Batting.csv")
pitching_data = pd.read_csv("../Data/Pitching.csv")
fielding_data = pd.read_csv("../Data/Fielding.csv")

batting_data.replace([np.inf, -np.inf], np.nan, inplace=True)
batting_data = batting_data.fillna(0)
pitching_data.replace([np.inf, -np.inf], np.nan, inplace=True)
pitching_data = pitching_data.fillna(0)
fielding_data.replace([np.inf, -np.inf], np.nan, inplace=True)
fielding_data = fielding_data.fillna(0)

batting_metrics = ["AB", "R", "H", "2B", "3B", "HR", "RBI", "SB", "CS", "BB", "SO", "IBB", "HBP", "SH", "SF", "GIDP"]
batting_career = batting_data.groupby("playerID")[batting_metrics].sum().reset_index()

batting_career["BA"] = batting_career["H"] / batting_career["AB"]
batting_career["OBP"] = batting_career["H"] + batting_career["BB"] + batting_career["HBP"] / batting_career["AB"] + batting_career["BB"] + batting_career["HBP"] + batting_career["SF"]

batting_career["SLG"] = batting_career["H"] + batting_career["2B"] + 2*batting_career["3B"] + 3*batting_career["HR"] / batting_career["AB"]

batting_career["OPS"] = batting_career["OBP"] + batting_career["SLG"]

pitching_metrics = ["W", "L", "G", "GS", "CG", "SHO", "SV", "IPouts", "H", "ER", "HR", "BB", "SO", "BAOpp", "ERA", "IBB", "WP", "HBP", "BK", "BFP", "GF", "R", "SH", "SF", "GIDP"]
pitching_career = pitching_data.groupby("playerID")[pitching_metrics].sum().reset_index()

pitching_career["ERA"] = 9 * (pitching_career["ER"] / pitching_career["IPouts"])
pitching_career["WHIP"] = 3 * (pitching_career["H"] + pitching_career["BB"] / pitching_career["IPouts"])

fielding_metrics = ["G", "GS", "InnOuts", "PO", "A", "E", "DP", "PB", "WP", "SB", "CS", "ZR"]
fielding_career = fielding_data.groupby("playerID")[fielding_metrics].sum().reset_index()

batting_data.replace([np.inf, -np.inf], np.nan, inplace=True)
batting_data = batting_data.fillna(0)
pitching_data.replace([np.inf, -np.inf], np.nan, inplace=True)
pitching_data = pitching_data.fillna(0)
fielding_data.replace([np.inf, -np.inf], np.nan, inplace=True)
fielding_data = fielding_data.fillna(0)

fielding_batting_career = pd.merge(batting_career, fielding_career, on="playerID", how="outer")

hof = pd.read_csv("../Data/HallOfFame.csv")


pitching_career["HOF"] = 0
fielding_batting_career["HOF"] = 0

hof_players = hof[hof["inducted"] == "Y"]["playerID"].unique()
pitching_career.loc[pitching_career["playerID"].isin(hof_players), "HOF"] = 1
fielding_batting_career.loc[fielding_batting_career["playerID"].isin(hof_players), "HOF"] = 1

pitching_career.to_csv("../Data/MasterPitching.csv", index=False)
fielding_batting_career.to_csv("../Data/MasterNonPitching.csv", index=False)



