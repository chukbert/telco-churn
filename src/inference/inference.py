#!/usr/bin/env python3
"""
SageMaker Inference Handler for Telco Customer Churn Prediction
Supports both Naive Bayes and TensorFlow DNN models
"""

import json
import os
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow import keras
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChurnPredictor:
    """Handles model loading and predictions for churn models"""
    
    def __init__(self, model_dir="/opt/ml/model"):
        self.model_dir = model_dir
        self.scaler = None
        self.feature_names = None
        self.tf_model = None
        self.nb_model = None
        self.best_model_type = None
        
        # Load models and artifacts
        self._load_artifacts()
    
    def _load_artifacts(self):
        """Load all necessary artifacts"""
        logger.info(f"Loading artifacts from {self.model_dir}")
        
        try:
            # Load scaler
            scaler_path = os.path.join(self.model_dir, 'scaler.pkl')
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
                logger.info("Scaler loaded successfully")
            else:
                logger.warning("Scaler not found")
            
            # Load feature names
            feature_names_path = os.path.join(self.model_dir, 'feature_names.json')
            if os.path.exists(feature_names_path):
                with open(feature_names_path, 'r') as f:
                    self.feature_names = json.load(f)
                logger.info(f"Feature names loaded: {len(self.feature_names)} features")
            else:
                logger.warning("Feature names not found")
            
            # Load training results to determine best model
            results_path = os.path.join(self.model_dir, 'training_results.json')
            if os.path.exists(results_path):
                with open(results_path, 'r') as f:
                    results = json.load(f)
                    self.best_model_type = results.get('best_model', 'tensorflow_dnn')
                logger.info(f"Best model type: {self.best_model_type}")
            else:
                self.best_model_type = 'tensorflow_dnn'  # Default
                logger.warning("Training results not found, defaulting to TensorFlow DNN")
            
            # Load TensorFlow model
            tf_model_path = os.path.join(self.model_dir, 'tensorflow_dnn_model.keras')
            if os.path.exists(tf_model_path):
                self.tf_model = keras.models.load_model(tf_model_path)
                logger.info("TensorFlow DNN model loaded successfully")
            else:
                # Try alternative path
                alt_tf_path = os.path.join(self.model_dir, 'model.keras')
                if os.path.exists(alt_tf_path):
                    self.tf_model = keras.models.load_model(alt_tf_path)
                    logger.info("TensorFlow DNN model loaded from alternative path")
                else:
                    logger.warning("TensorFlow DNN model not found")
            
            # Load Naive Bayes model
            nb_model_path = os.path.join(self.model_dir, 'naive_bayes_model.pkl')
            if os.path.exists(nb_model_path):
                self.nb_model = joblib.load(nb_model_path)
                logger.info("Naive Bayes model loaded successfully")
            else:
                logger.warning("Naive Bayes model not found")
            
        except Exception as e:
            logger.error(f"Error loading artifacts: {e}")
            raise
    
    def preprocess_input(self, input_data):
        """Preprocess input data to match training format"""
        try:
            # Convert to DataFrame if it's a dict
            if isinstance(input_data, dict):
                df = pd.DataFrame([input_data])
            elif isinstance(input_data, list):
                df = pd.DataFrame(input_data)
            else:
                df = input_data.copy()
            
            logger.info(f"Input shape: {df.shape}")
            logger.info(f"Input columns: {list(df.columns)}")
            
            # Apply the same feature engineering as in training
            df = self._apply_feature_engineering(df)
            
            # Ensure we have the right features
            if self.feature_names:
                # Add missing features with default values
                for feature in self.feature_names:
                    if feature not in df.columns:
                        df[feature] = 0
                        logger.warning(f"Missing feature {feature}, filled with 0")
                
                # Select and order features to match training
                df = df[self.feature_names]
            
            # Scale features
            if self.scaler:
                X_scaled = self.scaler.transform(df)
                logger.info("Features scaled successfully")
            else:
                X_scaled = df.values
                logger.warning("No scaler available, using raw features")
            
            return X_scaled
            
        except Exception as e:
            logger.error(f"Error in preprocessing: {e}")
            raise
    
    def _apply_feature_engineering(self, df):
        """Apply the same feature engineering as in training"""
        # Handle column name standardization
        if 'tenure' in df.columns:
            df = df.rename(columns={'tenure': 'Tenure Months'})
        if 'MonthlyCharges' in df.columns:
            df = df.rename(columns={'MonthlyCharges': 'Monthly Charges'})
        if 'TotalCharges' in df.columns:
            df = df.rename(columns={'TotalCharges': 'Total Charges'})
        
        # Convert TotalCharges to numeric
        if 'Total Charges' in df.columns:
            df['Total Charges'] = pd.to_numeric(df['Total Charges'], errors='coerce')
        
        # Fill missing values
        if 'Total Charges' in df.columns and df['Total Charges'].isnull().sum() > 0:
            df['Total Charges'].fillna(df['Monthly Charges'], inplace=True)
        
        # Remove target-related columns if present
        target_columns = ['Churn Value', 'Churn', 'Churn Label', 'Churn Reason']
        for col in target_columns:
            if col in df.columns:
                df = df.drop(col, axis=1)
        
        # Remove ID columns
        id_columns = [col for col in df.columns if 'customer' in col.lower() or 'id' in col.lower()]
        if id_columns:
            df = df.drop(id_columns, axis=1)
        
        # Identify categorical and numerical features
        categorical_features = df.select_dtypes(include=['object']).columns.tolist()
        numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        # Advanced feature engineering for numerical features
        if 'Monthly Charges' in numerical_features and 'Tenure Months' in numerical_features:
            # Total value as a customer
            df['Total_Value'] = df['Monthly Charges'] * df['Tenure Months']
            
            # Average monthly spending (handling division by zero)
            if 'Total Charges' in df.columns:
                df['Avg_Monthly_Value'] = df['Total Charges'] / (df['Tenure Months'] + 1)
            
            # Tenure categories
            df['Tenure_Category'] = pd.cut(df['Tenure Months'], 
                                         bins=[0, 12, 24, 48, 100], 
                                         labels=['New', 'Medium', 'Long', 'Veteran'])
            
            # Monthly charges categories
            df['Charges_Category'] = pd.cut(df['Monthly Charges'], 
                                          bins=[0, 35, 65, 100], 
                                          labels=['Low', 'Medium', 'High'])
        
        # Handle categorical encoding (simplified for inference)
        categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
        for col in categorical_features:
            if col in df.columns:
                # Simple label encoding for inference
                # In production, you'd want to save and load the exact encoders from training
                unique_vals = df[col].unique()
                encoding_map = {val: i for i, val in enumerate(unique_vals)}
                df[col + '_Encoded'] = df[col].map(encoding_map).fillna(0)
                df = df.drop(col, axis=1)
        
        return df
    
    def predict(self, input_data):
        """Make predictions using the best model"""
        try:
            # Preprocess input
            X_processed = self.preprocess_input(input_data)
            
            # Choose model based on best_model_type
            if self.best_model_type == 'tensorflow_dnn' and self.tf_model is not None:
                # TensorFlow DNN prediction
                predictions_prob = self.tf_model.predict(X_processed, verbose=0)[:, 0]
                predictions_binary = (predictions_prob > 0.5).astype(int)
                model_used = 'TensorFlow DNN'
                
            elif self.best_model_type == 'naive_bayes' and self.nb_model is not None:
                # Naive Bayes prediction
                predictions_prob = self.nb_model.predict_proba(X_processed)[:, 1]
                predictions_binary = self.nb_model.predict(X_processed)
                model_used = 'Naive Bayes'
                
            else:
                # Fallback to any available model
                if self.tf_model is not None:
                    predictions_prob = self.tf_model.predict(X_processed, verbose=0)[:, 0]
                    predictions_binary = (predictions_prob > 0.5).astype(int)
                    model_used = 'TensorFlow DNN (fallback)'
                elif self.nb_model is not None:
                    predictions_prob = self.nb_model.predict_proba(X_processed)[:, 1]
                    predictions_binary = self.nb_model.predict(X_processed)
                    model_used = 'Naive Bayes (fallback)'
                else:
                    raise ValueError("No models available for prediction")
            
            # Format results
            results = []
            for i in range(len(predictions_prob)):
                result = {
                    'churn_probability': float(predictions_prob[i]),
                    'churn_prediction': int(predictions_binary[i]),
                    'churn_label': 'Yes' if predictions_binary[i] == 1 else 'No',
                    'model_used': model_used,
                    'confidence': 'High' if abs(predictions_prob[i] - 0.5) > 0.3 else 'Medium' if abs(predictions_prob[i] - 0.5) > 0.1 else 'Low'
                }
                results.append(result)
            
            logger.info(f"Predictions made using {model_used}")
            return results
            
        except Exception as e:
            logger.error(f"Error in prediction: {e}")
            raise

# Global predictor instance
predictor = None

def model_fn(model_dir):
    """Load model for SageMaker inference"""
    global predictor
    logger.info(f"Loading model from {model_dir}")
    predictor = ChurnPredictor(model_dir)
    return predictor

def input_fn(request_body, request_content_type):
    """Parse input data for SageMaker inference"""
    logger.info(f"Processing input with content type: {request_content_type}")
    
    if request_content_type == 'application/json':
        input_data = json.loads(request_body)
    elif request_content_type == 'text/csv':
        # Handle CSV input
        import io
        df = pd.read_csv(io.StringIO(request_body))
        input_data = df.to_dict('records')
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")
    
    return input_data

def predict_fn(input_data, model):
    """Make predictions using the loaded model"""
    logger.info("Making predictions")
    return model.predict(input_data)

def output_fn(prediction, content_type):
    """Format prediction output for SageMaker inference"""
    logger.info(f"Formatting output with content type: {content_type}")
    
    if content_type == 'application/json':
        return json.dumps(prediction)
    else:
        raise ValueError(f"Unsupported content type: {content_type}")

# For local testing
if __name__ == "__main__":
    # Test the predictor locally
    predictor = ChurnPredictor("../../models")
    
    # Sample test data
    test_data = {
        "Gender": "Female",
        "Senior Citizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "Tenure Months": 1,
        "Phone Service": "No",
        "Multiple Lines": "No phone service",
        "Internet Service": "DSL",
        "Online Security": "No",
        "Online Backup": "Yes",
        "Device Protection": "No",
        "Tech Support": "No",
        "Streaming TV": "No",
        "Streaming Movies": "No",
        "Contract": "Month-to-month",
        "Paperless Billing": "Yes",
        "Payment Method": "Electronic check",
        "Monthly Charges": 29.85,
        "Total Charges": 29.85
    }
    
    try:
        result = predictor.predict(test_data)
        print("Test prediction successful:")
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"Test failed: {e}")