import fastf1 as ff1
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Enable cache directory
ff1.Cache.enable_cache('cache')

# Gather 2024 race results up to the Canadian GP
schedule_2024 = ff1.get_event_schedule(2024)
completed_2024 = schedule_2024[schedule_2024['EventDate'] < pd.Timestamp.now()]

data_rows = []
for _, event in completed_2024.iterrows():
    try:
        session = ff1.get_session(event['EventDate'].year, event['EventName'], 'R')
        session.load()
        for _, r in session.results.iterrows():
            data_rows.append({
                'Abbreviation': r['Abbreviation'],
                'TeamName': r['TeamName'],
                'GridPosition': r['GridPosition'],
                'Circuit': event['EventName'],
                'Year': event['EventDate'].year,
                'Position': r['Position'],
            })
    except Exception as e:
        print(f"Skipping {event['EventName']}: {e}")

# Build dataframe
race_df = pd.DataFrame(data_rows)
# Target variable: did the driver win?
race_df['Won'] = race_df['Position'].apply(lambda x: 1 if x == 1 else 0)

# Encode categorical variables
label_encoders = {}
for col in ['Abbreviation', 'TeamName', 'Circuit']:
    le = LabelEncoder()
    race_df[col] = le.fit_transform(race_df[col])
    label_encoders[col] = le

features = ['Abbreviation', 'TeamName', 'GridPosition', 'Circuit', 'Year']
X = race_df[features]
y = race_df['Won']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)
print("Train accuracy:", accuracy_score(y_train, model.predict(X_train)))
print("Test accuracy:", accuracy_score(y_test, model.predict(X_test)))

# Prepare qualifying data for Canadian GP 2025
try:
    qual_session = ff1.get_session(2025, 'Canadian Grand Prix', 'Q')
    qual_session.load()
except Exception:
    # If 2025 data isn't available, use 2024 qualifying
    qual_session = ff1.get_session(2024, 'Canadian Grand Prix', 'Q')
    qual_session.load()

qual_results = qual_session.results[['Abbreviation', 'TeamName', 'GridPosition']].copy()
qual_results['Circuit'] = 'Canadian Grand Prix'
qual_results['Year'] = 2025

# Apply encoders; handle unseen labels
for col in ['Abbreviation', 'TeamName', 'Circuit']:
    le = label_encoders[col]
    qual_results[col] = qual_results[col].map(lambda s: 'Unknown' if s not in le.classes_ else s)
    if 'Unknown' not in le.classes_:
        le.classes_ = list(le.classes_) + ['Unknown']
    qual_results[col] = le.transform(qual_results[col])

X_canada = qual_results[features]
qual_results['WinProbability'] = model.predict_proba(X_canada)[:, 1]

print("\nPredicted win probabilities for Canadian GP 2025:")
print(qual_results.sort_values('WinProbability', ascending=False)[['Abbreviation', 'TeamName', 'GridPosition', 'WinProbability']])
