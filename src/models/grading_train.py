import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from src.models.grading import extract_features

# Load the new MedQuAD dataset
DATA_PATH = 'data/medquad_grading_data.csv'
df = pd.read_csv(DATA_PATH)
print(f"Loaded {len(df)} grading examples from MedQuAD dataset")

# Feature extraction
feature_rows = []
labels = []
for i, row in df.iterrows():
    if i % 1000 == 0:  # Progress indicator
        print(f"Processing example {i}/{len(df)}")
    try:
        feats = extract_features(row['question'], row['reference_answer'], row['user_answer'], row['lang'])
        feature_rows.append([feats[k] for k in sorted(feats)])
        labels.append(row['score'])
    except Exception as e:
        print(f"Error processing example {i}: {e}")
        continue

X = feature_rows
y = labels
print(f"Extracted features for {len(X)} examples")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Evaluate
score = rf.score(X_test, y_test)
print(f"Test R^2: {score:.3f}")

# Save model
joblib.dump(rf, 'data/grading_rf_model.pkl')
print("Model saved to data/grading_rf_model.pkl")

