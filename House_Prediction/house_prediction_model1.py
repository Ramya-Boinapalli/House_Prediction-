import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

# Load dataset (replace with your dataset path)
df = pd.read_csv(r"C:\Users\Ramya\VS Code Project\Streamlit\DS10AM\ml\House_data.csv")

# Select features & target
X = df[["sqft_living", "bedrooms", "bathrooms"]]  # features
y = df["price"]                                   # target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Save model
with open("house_price_model.pkl", "wb") as f:
    pickle.dump(model, f)

print(" Model trained & saved as house_price_model.pkl")