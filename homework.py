import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
url = "https://aegis4048.github.io/downloads/notebooks/sample_data/unconv_MV_v5.csv"
df = pd.read_csv(url)
features = ['Por', 'Brittle', 'Perm', 'TOC']
target = 'Prod'
X = df[features]
y = pd.qcut(df[target], q=3, labels=['Low', 'Medium', 'High'])
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Actual values:", y_test.head(10).tolist())
print("Predicted values:", y_pred[:10].tolist())
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", round(accuracy * 100, 2), "%")
new_data = pd.DataFrame({
    'Por': [12.0],
    'Brittle': [45.0],
    'Perm': [100.0],
    'TOC': [2.5]
})
prediction = model.predict(new_data)
print("Predicted Production Class:", prediction[0])