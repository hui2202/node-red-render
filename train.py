import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score
import joblib

# Load data
column_names = ['date_time', 'traffic_volume', 'temp', 'rain_1h', 'hour', 'is_peak', 'day_of_week', 'traffic_level']
df = pd.read_csv('cleaned_output.csv', names=column_names)

# Features: temp, rain_1h, hour, is_peak, day_of_week (5 features)
X = df[['temp', 'rain_1h', 'hour', 'is_peak', 'day_of_week']]
y = df['traffic_level']

# Split: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
model = DecisionTreeClassifier(max_depth=15, min_samples_leaf=100, random_state=42)
model.fit(X_train, y_train)

# Evaluation - Save these results for the report
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')

# --- ML Evaluation ---
print(f"--- Model Results ---")
print(f"Accuracy: {acc:.2f}")
print(f"F1 Score: {f1:.2f}")

# --- Baseline Accuracy ---
print("\n--- Baseline Results (Most Frequent Class) ---")

# Step 1: Identify the most frequent class in your training labels
most_frequent_label = y_train.value_counts().idxmax()

# Step 2: Create a list of 'guesses' that are all the most frequent label
baseline_preds = [most_frequent_label] * len(y_test)

# Step 3: Calculate metrics for the baseline
baseline_acc = accuracy_score(y_test, baseline_preds)
baseline_f1 = f1_score(y_test, baseline_preds, average='weighted')

print(f"Baseline Accuracy: {baseline_acc:.2f}")
print(f"Baseline F1 Score: {baseline_f1:.2f}")

# Save the model
joblib.dump(model, 'traffic_model.pkl')
print("File 'traffic_model.pkl' created.")