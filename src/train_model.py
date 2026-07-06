import os
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from data_preprocessing import load_and_preprocess_data

def train_and_save_models():
    # Set up paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, '..', 'data', 'Concrete_Data.xls')
    models_dir = os.path.join(current_dir, '..', 'models')
    
    # Ensure models directory exists
    os.makedirs(models_dir, exist_ok=True)
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data(data_path)
    
    # Save the scaler for future predictions
    scaler_path = os.path.join(models_dir, 'scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved to {scaler_path}")
    
    # Initialize models
    models = {
        'linear_reg': LinearRegression(),
        'random_forest': RandomForestRegressor(n_estimators=100, random_state=42)
    }
    
    # Train, evaluate, and save models
    for model_name, model in models.items():
        print(f"\n--- Training {model_name} ---")
        model.fit(X_train, y_train)
        
        # Evaluate on the test set
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"MSE: {mse:.2f}")
        print(f"R2 Score: {r2:.4f}")
        
        # Save model
        model_path = os.path.join(models_dir, f'{model_name}.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train_and_save_models()
