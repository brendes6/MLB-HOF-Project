from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

pitching_df = pd.read_csv("../Data/MasterPitching.csv")
metrics = ["W", "L", "G", "GS", "CG", "SHO", "SV", "IPouts",
    "H", "ER", "HR", "BB", "SO", "BAOpp", "ERA", "IBB",
    "WP", "HBP", "BK", "BFP", "GF", "R", "SH", "SF",
    "GIDP", "WHIP"]


X = pitching_df[metrics]
y = pitching_df["HOF"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
model = LogisticRegression(max_iter=1000, class_weight="balanced")
model.fit(X_train, y_train)

cv_scores = cross_val_score(model, X_train, y_train, cv=5)
print(f"Cross-validation scores: {cv_scores}")
print(f"Average CV score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

y_pred = model.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

feature_importance = pd.DataFrame({
    'Feature': metrics,
    'Importance': abs(model.coef_[0])
})
feature_importance = feature_importance.sort_values('Importance', ascending=False)
print("\nFeature Importance:")
print(feature_importance)
feature_importance.plot(x="Feature", y="Importance", kind="bar")
plt.show()

joblib.dump(model, "../Models/pitching_model.pkl")
joblib.dump(scaler, "../Scalers/pitching_scaler.pkl")



