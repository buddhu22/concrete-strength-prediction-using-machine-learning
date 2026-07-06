import os
import pickle
import numpy as np
import pandas as pd

def predict_strength(features, model_name='random_forest'):
    """
    Predict concrete compressive strength based on input features.
    
    Features should be a list or dict containing 8 values corresponding to:
    Cement, Blast Furnace Slag, Fly Ash, Water, Superplasticizer, 
    Coarse Aggregate, Fine Aggregate, Age
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(current_dir, '..', 'models')
    
    scaler_path = os.path.join(models_dir, 'scaler.pkl')
    model_path = os.path.join(models_dir, f'{model_name}.pkl')
    
    if not os.path.exists(scaler_path) or not os.path.exists(model_path):
        raise FileNotFoundError("Scaler or Model file is missing. Please train the model first.")
        
    # Load scaler and model
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
        
    # Feature columns used during training
    feature_names = ['Cement (component 1)(kg in a m^3 mixture)', 
                     'Blast Furnace Slag (component 2)(kg in a m^3 mixture)', 
                     'Fly Ash (component 3)(kg in a m^3 mixture)', 
                     'Water  (component 4)(kg in a m^3 mixture)', 
                     'Superplasticizer (component 5)(kg in a m^3 mixture)', 
                     'Coarse Aggregate  (component 6)(kg in a m^3 mixture)', 
                     'Fine Aggregate (component 7)(kg in a m^3 mixture)', 
                     'Age (day)']
                     
    if isinstance(features, (list, np.ndarray)):
        # If it's a list, assume order matches training data
        features_df = pd.DataFrame([features], columns=feature_names)
    elif isinstance(features, dict):
        # Map simple dictionary keys to correct column names if necessary,
        # but for simplicity, we assume dict keys are aligned or we just use list
        features_df = pd.DataFrame([features])
    else:
        raise ValueError("Features must be a list, numpy array, or dictionary.")
        
    # Scale features
    features_scaled = scaler.transform(features_df)
    features_scaled_df = pd.DataFrame(features_scaled, columns=feature_names)
    
    # Predict
    prediction = model.predict(features_scaled_df)
    
    return prediction[0]

if __name__ == "__main__":
    # Example sample: [Cement, Slag, Fly Ash, Water, Superplasticizer, Coarse Aggregate, Fine Aggregate, Age]
    sample_features = [540.0, 0.0, 0.0, 162.0, 2.5, 1040.0, 676.0, 28]
    
    print("Testing prediction for sample features:")
    print(sample_features)
    
    try:
        rf_pred = predict_strength(sample_features, model_name='random_forest')
        print(f"\nRandom Forest Prediction: {rf_pred:.2f} MPa")
        
        lr_pred = predict_strength(sample_features, model_name='linear_reg')
        print(f"Linear Regression Prediction: {lr_pred:.2f} MPa")
    except Exception as e:
        print(f"Error during prediction: {e}")
