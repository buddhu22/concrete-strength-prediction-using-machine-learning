import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

def load_and_preprocess_data(file_path):
    """
    Loads the concrete dataset and performs preprocessing:
    - Splits data into features (X) and target (y)
    - Performs train-test split
    - Scales the features using StandardScaler
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at {file_path}")

    # Load dataset
    df = pd.read_excel(file_path)

    # The last column is the target variable (Concrete compressive strength)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Split the dataset into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the features using StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert back to DataFrame to preserve column names
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

if __name__ == "__main__":
    # Test the preprocessing script
    # Path relative to the src directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, '..', 'data', 'Concrete_Data.xls')
    
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data(data_path)
    print(f"Data preprocessing successful!")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
