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
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Evaluation - Save these results for the report
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"--- Model Results ---")
print(f"Accuracy: {acc:.2f}")
print(f"F1 Score: {f1:.2f}")

# Save the model
joblib.dump(model, 'traffic_model.pkl')
print("File 'traffic_model.pkl' created.")