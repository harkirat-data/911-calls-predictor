import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import pickle

# Load data
df = pd.read_csv('911.csv')

# Extract Reason from title column (e.g. "EMS: Cardiac Emergency" → "EMS")
df['Reason'] = df['title'].apply(lambda x: x.split(':')[0])

# Preprocessing
df['timeStamp'] = pd.to_datetime(df['timeStamp'])
df['hour']        = df['timeStamp'].dt.hour
df['month']       = df['timeStamp'].dt.month
df['day of week'] = df['timeStamp'].dt.dayofweek
df['year']        = df['timeStamp'].dt.year

# Encode target
le_reason = LabelEncoder()
df['Reason_enc'] = le_reason.fit_transform(df['Reason'])

print("Classes found:", le_reason.classes_)

# Features
features = ['lat', 'lng', 'day of week', 'month', 'hour', 'year']
X = df[features]
y = df['Reason_enc']

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                    test_size=0.2, random_state=42)

# Train Random Forest
rf = RandomForestClassifier(n_estimators=100, max_depth=10,
                             random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

# Save everything
pickle.dump(rf,        open('model.pkl', 'wb'))
pickle.dump(le_reason, open('le_reason.pkl', 'wb'))

print("✅ Model saved successfully!")