#!/usr/bin/env python3
"""
SageMaker Training Script for Telco Customer Churn Prediction
Trains both Naive Bayes and TensorFlow DNN models for comparison
"""

import argparse
import json
import os
import sys
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score, 
    precision_score, recall_score, classification_report
)
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, mixed_precision
import joblib
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_gpu():
    """Setup GPU and mixed precision if available"""
    logger.info("Setting up GPU configuration...")
    
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        logger.info(f"GPU found: {gpus[0].name}")
        try:
            # Enable memory growth
            tf.config.experimental.set_memory_growth(gpus[0], True)
            # Enable mixed precision
            mixed_precision.set_global_policy('mixed_float16')
            logger.info("Mixed precision enabled (FP16)")
        except Exception as e:
            logger.warning(f"GPU setup failed: {e}")
    else:
        logger.info("No GPU found, using CPU")
    
    logger.info(f"TensorFlow version: {tf.__version__}")

def load_and_preprocess_data(input_path):
    """Load and preprocess the telco customer churn data"""
    logger.info(f"Loading data from {input_path}")
    
    # Try different file formats
    data_file = None
    for filename in os.listdir(input_path):
        if filename.endswith(('.xlsx', '.csv')):
            data_file = os.path.join(input_path, filename)
            break
    
    if not data_file:
        raise FileNotFoundError(f"No Excel or CSV file found in {input_path}")
    
    # Load data
    if data_file.endswith('.xlsx'):
        df = pd.read_excel(data_file)
    else:
        df = pd.read_csv(data_file)
    
    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"Columns: {list(df.columns)}")
    
    # Create proper target variable
    if 'Churn Value' in df.columns:
        if df['Churn Value'].dtype in ['int64', 'float64']:
            logger.info(f"Churn rate: {(df['Churn Value'] == 1).mean():.2%}")
        else:
            df['Churn Value'] = df['Churn Value'].map({'Yes': 1, 'No': 0})
            logger.info(f"Churn rate: {(df['Churn Value'] == 1).mean():.2%}")
    elif 'Churn Label' in df.columns:
        df['Churn Value'] = df['Churn Label'].map({'Yes': 1, 'No': 0})
        logger.info(f"Churn rate: {(df['Churn Value'] == 1).mean():.2%}")
    elif 'Churn' in df.columns:
        if df['Churn'].dtype == 'object':
            df['Churn Value'] = df['Churn'].map({'Yes': 1, 'No': 0})
        else:
            df['Churn Value'] = df['Churn']
        logger.info(f"Churn rate: {(df['Churn Value'] == 1).mean():.2%}")
    else:
        raise ValueError("No suitable churn column found")
    
    # Verify target variable
    logger.info(f"Target variable distribution: {df['Churn Value'].value_counts().to_dict()}")
    if df['Churn Value'].nunique() == 1:
        raise ValueError("All target values are the same - no variation in target variable")
    
    # Data cleaning
    logger.info("Starting data preprocessing...")
    
    # Drop customer ID columns
    id_columns = [col for col in df.columns if 'customer' in col.lower() or 'id' in col.lower()]
    if id_columns:
        df = df.drop(id_columns, axis=1)
        logger.info(f"Dropped ID columns: {id_columns}")
    
    # Handle TotalCharges conversion
    if 'Total Charges' in df.columns:
        df['Total Charges'] = pd.to_numeric(df['Total Charges'], errors='coerce')
    elif 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df = df.rename(columns={'TotalCharges': 'Total Charges'})
    
    # Standardize column names
    if 'tenure' in df.columns:
        df = df.rename(columns={'tenure': 'Tenure Months'})
    if 'MonthlyCharges' in df.columns:
        df = df.rename(columns={'MonthlyCharges': 'Monthly Charges'})
    
    # Handle missing values
    logger.info(f"Missing values before cleaning: {df.isnull().sum().sum()}")
    
    if 'Total Charges' in df.columns and df['Total Charges'].isnull().sum() > 0:
        df['Total Charges'].fillna(df['Monthly Charges'], inplace=True)
        logger.info(f"Filled missing values in Total Charges")
    
    # Handle Churn Reason - fill with 'No Churn' for non-churners
    if 'Churn Reason' in df.columns:
        mask = df['Churn Reason'].isnull() & (df['Churn Value'] == 0)
        df.loc[mask, 'Churn Reason'] = 'No Churn'
        logger.info(f"Filled Churn Reason for non-churners")
    
    # Drop rows with critical missing values only
    critical_columns = ['Monthly Charges', 'Tenure Months']
    before_drop = df.shape[0]
    df = df.dropna(subset=[col for col in critical_columns if col in df.columns])
    after_drop = df.shape[0]
    logger.info(f"Dropped {before_drop - after_drop} rows with critical missing values")
    
    return df

def advanced_feature_engineering(df):
    """Perform advanced feature engineering"""
    logger.info("Performing advanced feature engineering...")
    
    # Separate features and target
    target_col = 'Churn Value'
    y = df[target_col].copy()
    X = df.drop([target_col], axis=1).copy()
    
    # Also drop any remaining churn-related columns
    churn_columns = [col for col in X.columns if 'churn' in col.lower()]
    if churn_columns:
        X = X.drop(churn_columns, axis=1)
        logger.info(f"Dropped additional churn columns: {churn_columns}")
    
    # Identify categorical and numerical features
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    logger.info(f"Categorical features: {len(categorical_features)}")
    logger.info(f"Numerical features: {len(numerical_features)}")
    
    # Advanced feature engineering for numerical features
    if 'Monthly Charges' in numerical_features and 'Tenure Months' in numerical_features:
        # Total value as a customer
        X['Total_Value'] = X['Monthly Charges'] * X['Tenure Months']
        
        # Average monthly spending (handling division by zero)
        X['Avg_Monthly_Value'] = X['Total Charges'] / (X['Tenure Months'] + 1)
        
        # Tenure categories
        X['Tenure_Category'] = pd.cut(X['Tenure Months'], 
                                     bins=[0, 12, 24, 48, 100], 
                                     labels=['New', 'Medium', 'Long', 'Veteran'])
        
        # Monthly charges categories
        X['Charges_Category'] = pd.cut(X['Monthly Charges'], 
                                      bins=[0, 35, 65, 100], 
                                      labels=['Low', 'Medium', 'High'])
        
        numerical_features.extend(['Total_Value', 'Avg_Monthly_Value'])
        categorical_features.extend(['Tenure_Category', 'Charges_Category'])
    
    # Target-guided ordinal encoding for categorical variables
    for col in categorical_features:
        if col in X.columns:
            # Calculate mean target for each category
            category_means = X.groupby(col)[col].count().to_dict()  # Use count as fallback
            try:
                # Try to calculate actual target means
                temp_df = pd.concat([X[col], y], axis=1)
                category_means = temp_df.groupby(col)[target_col].mean().to_dict()
            except:
                # Fallback to simple label encoding
                unique_vals = X[col].unique()
                category_means = {val: i for i, val in enumerate(unique_vals)}
            
            # Map categories to their mean target values
            X[col + '_Encoded'] = X[col].map(category_means)
            # Keep original for reference, but use encoded for modeling
            X = X.drop(col, axis=1)
    
    # Update feature lists
    categorical_features = [col for col in X.columns if col.endswith('_Encoded')]
    numerical_features = [col for col in X.columns if col not in categorical_features]
    
    logger.info(f"Final feature count: {X.shape[1]}")
    logger.info(f"Feature names: {list(X.columns)}")
    
    return X, y, numerical_features, categorical_features

def build_tensorflow_dnn(input_dim, dropout_rate=0.3):
    """Build TensorFlow DNN model"""
    logger.info(f"Building TensorFlow DNN with input dimension: {input_dim}")
    
    inputs = layers.Input(shape=(input_dim,))
    
    # First hidden layer
    x = layers.Dense(512, activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    
    # Second hidden layer
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    
    # Third hidden layer
    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate / 2)(x)
    
    # Fourth hidden layer
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(dropout_rate / 2)(x)
    
    # Output layer (keep in float32 for mixed precision)
    outputs = layers.Dense(1, activation='sigmoid', dtype='float32')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name='TensorFlow_DNN')
    return model

def train_models(X_train, X_val, X_test, y_train, y_val, y_test, model_dir):
    """Train both Naive Bayes and TensorFlow DNN models"""
    logger.info("Starting model training...")
    
    results = {}
    
    # 1. Train Naive Bayes (baseline)
    logger.info("Training Naive Bayes model...")
    nb_model = GaussianNB()
    nb_model.fit(X_train, y_train)
    
    # Evaluate Naive Bayes
    nb_pred_val = nb_model.predict_proba(X_val)[:, 1]
    nb_pred_test = nb_model.predict_proba(X_test)[:, 1]
    nb_pred_val_binary = nb_model.predict(X_val)
    nb_pred_test_binary = nb_model.predict(X_test)
    
    results['naive_bayes'] = {
        'val_auc': roc_auc_score(y_val, nb_pred_val),
        'test_auc': roc_auc_score(y_test, nb_pred_test),
        'val_accuracy': accuracy_score(y_val, nb_pred_val_binary),
        'test_accuracy': accuracy_score(y_test, nb_pred_test_binary),
        'val_f1': f1_score(y_val, nb_pred_val_binary, average='macro'),
        'test_f1': f1_score(y_test, nb_pred_test_binary, average='macro'),
        'val_precision': precision_score(y_val, nb_pred_val_binary, average='macro'),
        'test_precision': precision_score(y_test, nb_pred_test_binary, average='macro'),
        'val_recall': recall_score(y_val, nb_pred_val_binary, average='macro'),
        'test_recall': recall_score(y_test, nb_pred_test_binary, average='macro')
    }
    
    logger.info(f"Naive Bayes - Val AUC: {results['naive_bayes']['val_auc']:.4f}, Test AUC: {results['naive_bayes']['test_auc']:.4f}")
    
    # Save Naive Bayes model
    joblib.dump(nb_model, os.path.join(model_dir, 'naive_bayes_model.pkl'))
    
    # 2. Train TensorFlow DNN
    logger.info("Training TensorFlow DNN...")
    tf_model = build_tensorflow_dnn(X_train.shape[1])
    
    # Compile model
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    if mixed_precision.global_policy().name == 'mixed_float16':
        optimizer = mixed_precision.LossScaleOptimizer(optimizer)
    
    tf_model.compile(
        optimizer=optimizer,
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[
            keras.metrics.AUC(name='auc'),
            keras.metrics.BinaryAccuracy(name='accuracy'),
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall')
        ]
    )
    
    logger.info(f"TensorFlow DNN parameters: {tf_model.count_params():,}")
    
    # Calculate class weights
    neg_weight = len(y_train) / (2 * np.sum(y_train == 0))
    pos_weight = len(y_train) / (2 * np.sum(y_train == 1))
    class_weights = {0: neg_weight, 1: pos_weight}
    
    # Train the model
    history = tf_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=20,
        batch_size=64,
        class_weight=class_weights,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor='val_auc', patience=10, mode='max', restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6
            )
        ],
        verbose=1
    )
    
    # Evaluate TensorFlow DNN
    tf_pred_val = tf_model.predict(X_val, verbose=0)[:, 0]
    tf_pred_test = tf_model.predict(X_test, verbose=0)[:, 0]
    tf_pred_val_binary = (tf_pred_val > 0.5).astype(int)
    tf_pred_test_binary = (tf_pred_test > 0.5).astype(int)
    
    results['tensorflow_dnn'] = {
        'val_auc': roc_auc_score(y_val, tf_pred_val),
        'test_auc': roc_auc_score(y_test, tf_pred_test),
        'val_accuracy': accuracy_score(y_val, tf_pred_val_binary),
        'test_accuracy': accuracy_score(y_test, tf_pred_test_binary),
        'val_f1': f1_score(y_val, tf_pred_val_binary, average='macro'),
        'test_f1': f1_score(y_test, tf_pred_test_binary, average='macro'),
        'val_precision': precision_score(y_val, tf_pred_val_binary, average='macro'),
        'test_precision': precision_score(y_test, tf_pred_test_binary, average='macro'),
        'val_recall': recall_score(y_val, tf_pred_val_binary, average='macro'),
        'test_recall': recall_score(y_test, tf_pred_test_binary, average='macro')
    }
    
    logger.info(f"TensorFlow DNN - Val AUC: {results['tensorflow_dnn']['val_auc']:.4f}, Test AUC: {results['tensorflow_dnn']['test_auc']:.4f}")
    
    # Save TensorFlow model
    tf_model.save(os.path.join(model_dir, 'tensorflow_dnn_model.keras'))
    
    # Determine best model
    tf_auc = results['tensorflow_dnn']['test_auc']
    nb_auc = results['naive_bayes']['test_auc']
    
    if tf_auc > nb_auc:
        best_model = 'tensorflow_dnn'
        logger.info(f"Best model: TensorFlow DNN (AUC: {tf_auc:.4f} vs {nb_auc:.4f})")
        # Save the best model as the default model for inference
        tf_model.save(os.path.join(model_dir, 'model.keras'))
        results['best_model'] = 'tensorflow_dnn'
    else:
        best_model = 'naive_bayes'
        logger.info(f"Best model: Naive Bayes (AUC: {nb_auc:.4f} vs {tf_auc:.4f})")
        # For Naive Bayes, we'll use it in inference
        results['best_model'] = 'naive_bayes'
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Train Telco Churn Models')
    
    # SageMaker specific arguments
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR', '/opt/ml/model'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train'))
    parser.add_argument('--validation', type=str, default=os.environ.get('SM_CHANNEL_VALIDATION', '/opt/ml/input/data/validation'))
    parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST', '/opt/ml/input/data/test'))
    
    # Hyperparameters
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--dropout-rate', type=float, default=0.3)
    
    args = parser.parse_args()
    
    # Setup
    setup_gpu()
    
    # Load and preprocess data
    df = load_and_preprocess_data(args.train)
    X, y, numerical_features, categorical_features = advanced_feature_engineering(df)
    
    logger.info(f"Class distribution: {np.bincount(y)}")
    
    # Check if we have both classes
    if len(np.unique(y)) < 2:
        raise ValueError("Only one class found in target variable!")
    
    # Apply SMOTE to balance the dataset
    logger.info("Applying SMOTE for class balancing...")
    oversample = SMOTE(random_state=42)
    X_balanced, y_balanced = oversample.fit_resample(X, y)
    
    logger.info(f"Balanced class distribution: {np.bincount(y_balanced)}")
    logger.info(f"Original dataset size: {X.shape[0]}")
    logger.info(f"Balanced dataset size: {X_balanced.shape[0]}")
    
    # Split the balanced data
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_balanced, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.125, random_state=42, stratify=y_temp
    )
    
    logger.info(f"Data splits:")
    logger.info(f"Train: {X_train.shape} - Churn rate: {y_train.mean():.2%}")
    logger.info(f"Val:   {X_val.shape} - Churn rate: {y_val.mean():.2%}")
    logger.info(f"Test:  {X_test.shape} - Churn rate: {y_test.mean():.2%}")
    
    # Feature scaling
    logger.info("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Save the scaler
    joblib.dump(scaler, os.path.join(args.model_dir, 'scaler.pkl'))
    
    # Save feature names
    feature_names = list(X.columns)
    with open(os.path.join(args.model_dir, 'feature_names.json'), 'w') as f:
        json.dump(feature_names, f)
    
    # Train models
    results = train_models(X_train_scaled, X_val_scaled, X_test_scaled, 
                          y_train, y_val, y_test, args.model_dir)
    
    # Save results
    with open(os.path.join(args.model_dir, 'training_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info("Training completed successfully!")
    logger.info(f"Results saved to {args.model_dir}")
    
    # Print final summary
    logger.info("\n" + "="*50)
    logger.info("FINAL RESULTS SUMMARY")
    logger.info("="*50)
    for model_name, metrics in results.items():
        if model_name != 'best_model':
            logger.info(f"\n{model_name.upper()}:")
            logger.info(f"  Test AUC: {metrics['test_auc']:.4f}")
            logger.info(f"  Test Accuracy: {metrics['test_accuracy']:.4f}")
            logger.info(f"  Test F1: {metrics['test_f1']:.4f}")
    
    logger.info(f"\nBest model: {results['best_model']}")

if __name__ == '__main__':
    main()