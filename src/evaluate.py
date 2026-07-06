import os
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from data_preprocessing import load_and_preprocess_data

def evaluate_model(model_name='random_forest'):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, '..', 'data', 'Concrete_Data.xls')
    model_path = os.path.join(current_dir, '..', 'models', f'{model_name}.pkl')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
        
    print(f"Loading test data and {model_name} model...")
    _, X_test, _, y_test, _ = load_and_preprocess_data(data_path)
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
        
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n--- Evaluation Results for {model_name} ---")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"R-squared (R2): {r2:.4f}")
    print("-----------------------------------------")

if __name__ == "__main__":
    evaluate_model('random_forest')
    evaluate_model('linear_reg')
