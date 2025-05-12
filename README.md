# MLB Hall of Fame Prediction Project

A machine learning project that analyzes baseball statistics to predict Hall of Fame induction probability and identify key statistical factors that influence Hall of Fame voting.

## Project Overview

This project uses logistic regression to analyze MLB player statistics and predict Hall of Fame induction probability. It features:

- Separate prediction models for pitchers and non-pitchers
- Statistical significance analysis of different baseball metrics
- Interactive web interface for exploring predictions
- Comprehensive data processing pipeline for baseball statistics

## Technical Details

The project uses:
- Python with scikit-learn for logistic regression
- Streamlit for the web interface
- Pandas for data processing
- Standardized data scaling and preprocessing
- Cross-validation for model evaluation

## Local Setup

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit application:
   ```bash
   streamlit run src/app.py
   ```
4. Open your browser and navigate to the provided local URL (typically http://localhost:8501)

## Data Sources

The project uses the Baseball Databank dataset, which includes:
- Batting statistics
- Pitching statistics
- Fielding statistics
- Hall of Fame voting records
- Player biographical information

## Live Demo

https://mlb-project-brendes6.streamlit.app/

