import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
import joblib
import os

# === Load and preprocess ===
df = pd.read_csv("data/credit_risk_dataset.csv").dropna()

# Encode categorical variables
cat_cols = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']
for col in cat_cols:
    df[col] = LabelEncoder().fit_transform(df[col])

X = df.drop(columns='loan_status')
y = df['loan_status']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)

# Apply SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# === Train model with GridSearchCV ===
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5]
}
grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    scoring='f1',
    cv=3,
    n_jobs=-1
)
grid.fit(X_resampled, y_resampled)

# Evaluate
best_rf = grid.best_estimator_
y_pred = best_rf.predict(X_test)
print("Best parameters:", grid.best_params_)
print(classification_report(y_test, y_pred))

# === Save model ===
os.makedirs("models", exist_ok=True)
joblib.dump(best_rf, "models/rf_model.pkl")
