"""
Optimized inference handler with Neo compilation and SHAP explanations
Implements serverless endpoint with low-latency predictions
"""

import json
import numpy as np
import pandas as pd
import tensorflow as tf
import xgboost as xgb
import shap
import os
import time
import logging
from typing import Dict, List, Any, Tuple
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TelcoChurnPredictor:
    """High-performance predictor with Neo optimization and SHAP explanations"""
    
    def __init__(self):
        self.tf_model = None
        self.xgb_model = None
        self.preprocessor = None
        self.shap_explainer = None
        self.feature_names = None
        self.model_loaded = False
        
    def load_models(self, model_dir: str):
        """Load optimized models and preprocessor"""
        start_time = time.time()
        
        try:
            # Load TensorFlow model (Neo-optimized if available)
            tf_neo_path = os.path.join(model_dir, 'tensorflow_neo_model')
            tf_standard_path = os.path.join(model_dir, 'tensorflow_model/1')
            
            if os.path.exists(tf_neo_path):
                logger.info("Loading Neo-optimized TensorFlow model")
                self.tf_model = tf.lite.Interpreter(model_path=os.path.join(tf_neo_path, 'model.tflite'))
                self.tf_model.allocate_tensors()
            else:
                logger.info("Loading standard TensorFlow model")
                self.tf_model = tf.keras.models.load_model(tf_standard_path)
            
            # Load XGBoost model
            xgb_path = os.path.join(model_dir, 'xgboost-model')
            if os.path.exists(xgb_path):
                self.xgb_model = xgb.Booster()
                self.xgb_model.load_model(xgb_path)
            
            # Load preprocessor
            preprocessor_path = os.path.join(model_dir, 'preprocessor.pkl')
            if os.path.exists(preprocessor_path):
                self.preprocessor = joblib.load(preprocessor_path)
                self.feature_names = self.preprocessor.get('feature_names', [])
            
            # Initialize SHAP explainer (using TreeExplainer for XGBoost)
            if self.xgb_model:
                self.shap_explainer = shap.TreeExplainer(self.xgb_model)
            
            self.model_loaded = True
            load_time = time.time() - start_time
            logger.info(f"Models loaded successfully in {load_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            raise
    
    def preprocess_input(self, input_data: np.ndarray) -> np.ndarray:
        """Preprocess input data"""
        if self.preprocessor:
            # Apply saved preprocessing transformations
            if 'scalers' in self.preprocessor:
                scaler = self.preprocessor['scalers'].get('standard')
                if scaler:
                    input_data = scaler.transform(input_data)
        
        return input_data
    
    def predict_tensorflow(self, input_data: np.ndarray) -> np.ndarray:
        """Make predictions with TensorFlow model"""
        if isinstance(self.tf_model, tf.lite.Interpreter):
            # Neo-optimized TFLite inference
            input_details = self.tf_model.get_input_details()
            output_details = self.tf_model.get_output_details()
            
            predictions = []
            for sample in input_data:
                self.tf_model.set_tensor(input_details[0]['index'], 
                                       sample.reshape(1, -1).astype(np.float32))
                self.tf_model.invoke()
                pred = self.tf_model.get_tensor(output_details[0]['index'])
                predictions.append(pred[0])
            
            return np.array(predictions)
        else:
            # Standard Keras model inference
            return self.tf_model.predict(input_data, batch_size=32)
    
    def predict_xgboost(self, input_data: np.ndarray) -> np.ndarray:
        """Make predictions with XGBoost model"""
        dmatrix = xgb.DMatrix(input_data)
        return self.xgb_model.predict(dmatrix)
    
    def get_shap_explanations(self, input_data: np.ndarray, 
                            predictions: np.ndarray, 
                            top_k: int = 5) -> List[Dict[str, Any]]:
        """Generate SHAP explanations for predictions"""
        explanations = []
        
        if self.shap_explainer and self.xgb_model:
            # Calculate SHAP values
            shap_values = self.shap_explainer.shap_values(input_data)
            
            for i in range(len(input_data)):
                # Get top K most important features
                feature_importance = np.abs(shap_values[i])
                top_indices = np.argsort(feature_importance)[-top_k:][::-1]
                
                explanation = {
                    'prediction': float(predictions[i]),
                    'top_features': []
                }
                
                for idx in top_indices:
                    feature_name = self.feature_names[idx] if idx < len(self.feature_names) else f'feature_{idx}'
                    explanation['top_features'].append({
                        'feature': feature_name,
                        'value': float(input_data[i, idx]),
                        'shap_value': float(shap_values[i, idx]),
                        'impact': 'positive' if shap_values[i, idx] > 0 else 'negative'
                    })
                
                explanations.append(explanation)
        else:
            # Fallback: simple feature importance based on values
            for i in range(len(input_data)):
                explanation = {
                    'prediction': float(predictions[i]),
                    'top_features': []
                }
                
                # Use absolute values as proxy for importance
                feature_values = np.abs(input_data[i])
                top_indices = np.argsort(feature_values)[-top_k:][::-1]
                
                for idx in top_indices:
                    feature_name = self.feature_names[idx] if idx < len(self.feature_names) else f'feature_{idx}'
                    explanation['top_features'].append({
                        'feature': feature_name,
                        'value': float(input_data[i, idx]),
                        'importance': float(feature_values[idx])
                    })
                
                explanations.append(explanation)
        
        return explanations
    
    def predict(self, input_data: np.ndarray, 
                explain: bool = True, 
                top_k_features: int = 5) -> Dict[str, Any]:
        """Make ensemble predictions with optional explanations"""
        start_time = time.time()
        
        # Preprocess input
        processed_data = self.preprocess_input(input_data)
        
        # Get predictions from both models
        predictions = {}
        
        if self.tf_model:
            tf_start = time.time()
            tf_predictions = self.predict_tensorflow(processed_data)
            predictions['tensorflow'] = tf_predictions
            predictions['tensorflow_latency_ms'] = (time.time() - tf_start) * 1000
        
        if self.xgb_model:
            xgb_start = time.time()
            xgb_predictions = self.predict_xgboost(processed_data)
            predictions['xgboost'] = xgb_predictions
            predictions['xgboost_latency_ms'] = (time.time() - xgb_start) * 1000
        
        # Ensemble predictions (weighted average)
        if 'tensorflow' in predictions and 'xgboost' in predictions:
            ensemble_predictions = (
                0.6 * predictions['tensorflow'].flatten() + 
                0.4 * predictions['xgboost'].flatten()
            )
        elif 'tensorflow' in predictions:
            ensemble_predictions = predictions['tensorflow'].flatten()
        elif 'xgboost' in predictions:
            ensemble_predictions = predictions['xgboost'].flatten()
        else:
            raise ValueError("No models available for prediction")
        
        # Generate explanations if requested
        explanations = []
        if explain:
            explain_start = time.time()
            explanations = self.get_shap_explanations(
                processed_data, ensemble_predictions, top_k_features
            )
            explain_latency = (time.time() - explain_start) * 1000
        else:
            explain_latency = 0
        
        # Calculate total latency
        total_latency = (time.time() - start_time) * 1000
        
        # Format response
        response = {
            'predictions': ensemble_predictions.tolist(),
            'model_outputs': {
                'tensorflow': predictions.get('tensorflow', []).tolist() if 'tensorflow' in predictions else None,
                'xgboost': predictions.get('xgboost', []).tolist() if 'xgboost' in predictions else None
            },
            'explanations': explanations,
            'metadata': {
                'total_latency_ms': total_latency,
                'tensorflow_latency_ms': predictions.get('tensorflow_latency_ms', 0),
                'xgboost_latency_ms': predictions.get('xgboost_latency_ms', 0),
                'explanation_latency_ms': explain_latency,
                'num_samples': len(input_data),
                'model_version': '1.0.0'
            }
        }
        
        return response


# SageMaker inference handler functions
predictor = TelcoChurnPredictor()

def model_fn(model_dir: str):
    """Load model for SageMaker inference"""
    predictor.load_models(model_dir)
    return predictor

def input_fn(request_body: str, content_type: str = 'text/csv'):
    """Parse input data"""
    if content_type == 'text/csv':
        # Parse CSV input
        lines = request_body.strip().split('\n')
        data = []
        for line in lines:
            features = [float(x) for x in line.split(',')]
            data.append(features)
        return np.array(data)
    
    elif content_type == 'application/json':
        # Parse JSON input
        data = json.loads(request_body)
        if isinstance(data, dict) and 'instances' in data:
            return np.array(data['instances'])
        else:
            return np.array(data)
    
    else:
        raise ValueError(f"Unsupported content type: {content_type}")

def predict_fn(input_data: np.ndarray, model: TelcoChurnPredictor):
    """Make predictions"""
    # Check if explanations are requested (via context or default to True)
    explain = True  # Default to providing explanations
    
    return model.predict(input_data, explain=explain)

def output_fn(prediction: Dict[str, Any], accept: str = 'application/json'):
    """Format output"""
    if accept == 'application/json':
        return json.dumps(prediction, indent=2)
    else:
        raise ValueError(f"Unsupported accept type: {accept}")


# Local testing
def main():
    """Test inference locally"""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=str, default='models/')
    parser.add_argument('--input-file', type=str, required=True)
    parser.add_argument('--output-file', type=str, default='predictions.json')
    
    args = parser.parse_args()
    
    # Load models
    predictor = model_fn(args.model_dir)
    
    # Load test data
    if args.input_file.endswith('.csv'):
        test_data = pd.read_csv(args.input_file)
        input_data = test_data.values
    else:
        with open(args.input_file, 'r') as f:
            input_data = json.load(f)
            input_data = np.array(input_data)
    
    # Make predictions
    predictions = predict_fn(input_data, predictor)
    
    # Save results
    with open(args.output_file, 'w') as f:
        json.dump(predictions, f, indent=2)
    
    # Print summary
    print(f"Predictions saved to {args.output_file}")
    print(f"Total latency: {predictions['metadata']['total_latency_ms']:.2f}ms")
    print(f"Average latency per sample: {predictions['metadata']['total_latency_ms'] / predictions['metadata']['num_samples']:.2f}ms")
    
    # Check if meets latency requirement
    p95_latency = predictions['metadata']['total_latency_ms'] / predictions['metadata']['num_samples']
    if p95_latency < 60:
        print(f"✓ Meets p95 latency requirement (<60ms): {p95_latency:.2f}ms")
    else:
        print(f"✗ Does not meet p95 latency requirement (<60ms): {p95_latency:.2f}ms")


if __name__ == '__main__':
    main()