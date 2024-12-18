import pandas as pd

# Load raw financial data
raw_data = pd.read_csv('data/monthly_financial_data.csv')

# Preprocess data: Clean, remove duplicates, calculate profit
def preprocess_financial_data(df):
    df = df.drop_duplicates()
    df['Profit'] = df['Revenue'] - df['Expenses']
    return df

processed_data = preprocess_financial_data(raw_data)
processed_data.to_csv('data/processed_financial_data.csv', index=False)
print("ETL pipeline executed successfully. Processed data saved!")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Load preprocessed financial data
data = pd.read_csv('data/processed_financial_data.csv')

# Feature selection and target variable
X = data[['Month_Number']]  # Replace with actual features if available
y = data['Profit']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
print(f"Mean Absolute Error: {mae}")

# Save the model
import joblib
joblib.dump(model, 'models/budget_forecasting_model.pkl')

