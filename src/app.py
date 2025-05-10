import streamlit as st
from util import get_player_ID, get_player_name, pitcher, get_nonpitcher_stats, get_pitcher_stats, get_players, get_feature_significance_df
from predictions import predict_player
import random

st.markdown("## MLB Hall of Fame Probability & Stat Importance Explorer")

st.markdown("""
This app uses logistic regression models trained on Baseball Databank data to:
- Show which statistics are most influential for pitchers vs. non-pitchers
- Estimate a player’s probability of Hall of Fame induction based on career stats

You can input custom stats or explore stat importance below.
Note: The databank only contains/considers stats up to 2015.
""")

col1, col2 = st.columns(2)
    
with col1:
    st.markdown("**Stat Significance for Pitchers:**")
    pitching_features = get_feature_significance_df(True)
    st.pyplot(pitching_features.plot.bar(x="Feature", y="Importance").figure)
with col2:
    st.markdown("**Stat Significance for Non-Pitchers:**")
    nonpitching_features = get_feature_significance_df(False)
    st.pyplot(nonpitching_features.plot.bar(x="Feature", y="Importance").figure)


st.markdown("## Make Predictions")
player = st.selectbox("Please Select a Player:", get_players(), index=None, placeholder="Select a player")


if st.button("Predict HOF Probabilies") and player:
    preds, probs = predict_player(player)
    if preds[0] == 0:
        st.markdown(f"Player {player} is not predicted to make the hall of fame.")
    else:
        st.markdown(f"Player {player} is predicted to make the hall of fame.")

    st.markdown(f"Probability to make HOF: {probs[1]*100:.2f}%")

