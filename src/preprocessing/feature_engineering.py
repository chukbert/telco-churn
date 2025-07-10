"""
Feature engineering for Telco Customer Churn dataset
Implements advanced feature transformations optimized for SageMaker
Updated to handle XLSX format
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import logging
from typing import Dict, List, Tuple, Optional
import joblib
import json
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TelcoFeatureEngineer:
    """Feature engineering pipeline for Telco churn dataset"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._get_default_config()
        self.encoders = {}
        self.scalers = {}
        self.feature_names = []
        self.categorical_features = [
            'Gender', 'Senior Citizen', 'Partner', 'Dependents',
            'Phone Service', 'Multiple Lines', 'Internet Service',
            'Online Security', 'Online Backup', 'Device Protection',
            'Tech Support', 'Streaming TV', 'Streaming Movies',
            'Contract', 'Paperless Billing', 'Payment Method'
        ]
        self.numerical_features = [
            'Tenure Months', 'Monthly Charges', 'Total Charges', 'Churn Score', 'CLTV'
        ]
        
    def _get_default_config(self) -> Dict:
        return {
            'target_column': 'Churn Value',
            'test_size': 0.2,
            'validation_size': 0.1,
            'random_state': 42,
            'create_polynomial_features': True,
            'create_interaction_features': True,
            'handle_missing_values': True
        }
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load data from XLSX file"""
        logger.info(f"Loading data from {file_path}")
        
        # Check file extension and load accordingly
        if file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        elif file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
        
        logger.info(f"Loaded {len(df)} rows with {len(df.columns)} columns")
        logger.info(f"Columns: {list(df.columns)}")
        
        return df
    
    def create_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create advanced engineered features"""
        df = df.copy()
        
        # Revenue features
        df['AvgMonthlyCharges'] = df['Total Charges'] / (df['Tenure Months'] + 1)
        df['ChargesPerService'] = df['Monthly Charges'] / (
            df[['Phone Service', 'Internet Service', 'Streaming TV', 
                'Streaming Movies', 'Online Security', 'Online Backup', 
                'Device Protection', 'Tech Support']].apply(
                    lambda x: (x == 'Yes').sum(), axis=1
                ) + 1
        )
        
        # Contract features
        df['IsMonthToMonth'] = (df['Contract'] == 'Month-to-month').astype(int)
        df['HasLongTermContract'] = (df['Contract'].isin(['One year', 'Two year'])).astype(int)
        
        # Service bundle features
        service_columns = ['Phone Service', 'Internet Service', 'Streaming TV', 
                          'Streaming Movies', 'Online Security', 'Online Backup', 
                          'Device Protection', 'Tech Support']
        df['NumServices'] = df[service_columns].apply(lambda x: (x == 'Yes').sum(), axis=1)
        
        df['HasStreamingServices'] = (
            (df['Streaming TV'] == 'Yes') | (df['Streaming Movies'] == 'Yes')
        ).astype(int)
        
        df['HasSecurityServices'] = (
            (df['Online Security'] == 'Yes') | 
            (df['Online Backup'] == 'Yes') | 
            (df['Device Protection'] == 'Yes')
        ).astype(int)
        
        # Customer loyalty features
        df['TenureGroup'] = pd.cut(df['Tenure Months'], 
                                   bins=[0, 12, 24, 48, 72], 
                                   labels=['0-1yr', '1-2yr', '2-4yr', '4+yr'])
        
        # Payment features
        df['IsElectronicPayment'] = df['Payment Method'].str.contains('electronic').astype(int)
        df['IsAutoPayment'] = df['Payment Method'].str.contains('automatic').astype(int)
        
        # Interaction features
        if self.config['create_interaction_features']:
            df['SeniorWithDependents'] = (df['Senior Citizen'] == 'Yes').astype(int) * (df['Dependents'] == 'Yes').astype(int)
            df['MonthlyChargesXTenure'] = df['Monthly Charges'] * df['Tenure Months']
            df['NoInternetXCharges'] = (df['Internet Service'] == 'No').astype(int) * df['Monthly Charges']
        
        # Polynomial features for key numerical variables
        if self.config['create_polynomial_features']:
            df['tenure_squared'] = df['Tenure Months'] ** 2
            df['MonthlyCharges_squared'] = df['Monthly Charges'] ** 2
            df['tenure_log'] = np.log1p(df['Tenure Months'])
            df['TotalCharges_log'] = np.log1p(df['Total Charges'])
        
        return df
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset"""
        df = df.copy()
        
        # Log missing values
        missing_counts = df.isnull().sum()
        if missing_counts.sum() > 0:
            logger.info("Missing values found:")
            for col, count in missing_counts[missing_counts > 0].items():
                logger.info(f"  {col}: {count} missing values")
        
        # Total Charges has some missing values (new customers with tenure=0)
        if 'Total Charges' in df.columns:
            # Convert to numeric first (might be string with spaces)
            df['Total Charges'] = pd.to_numeric(df['Total Charges'], errors='coerce')
            # Fill missing with Monthly Charges (logical for new customers)
            df['Total Charges'].fillna(df['Monthly Charges'], inplace=True)
        
        # Handle categorical 'No internet service' and 'No phone service'
        for col in self.categorical_features:
            if col in df.columns:
                df[col] = df[col].replace({
                    'No internet service': 'No',
                    'No phone service': 'No'
                })
        
        # Drop customerID if exists (not useful for prediction)
        if 'customerID' in df.columns:
            df = df.drop('customerID', axis=1)
        
        return df
    
    def encode_categorical_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Encode categorical features"""
        df = df.copy()
        
        for col in self.categorical_features:
            if col in df.columns:
                if col == 'SeniorCitizen':
                    # Already binary (0/1)
                    continue
                    
                if fit:
                    # Fit and transform
                    encoder = LabelEncoder()
                    df[f'{col}_encoded'] = encoder.fit_transform(df[col])
                    self.encoders[col] = encoder
                else:
                    # Transform only
                    if col in self.encoders:
                        # Handle unseen categories
                        df[f'{col}_encoded'] = df[col].map(
                            lambda x: self.encoders[col].transform([x])[0] 
                            if x in self.encoders[col].classes_ else -1
                        )
                    else:
                        df[f'{col}_encoded'] = -1
        
        return df
    
    def scale_numerical_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Scale numerical features"""
        df = df.copy()
        
        # Extended list with engineered features
        numerical_cols = self.numerical_features + [
            'AvgMonthlyCharges', 'ChargesPerService', 'NumServices',
            'MonthlyChargesXTenure'
        ]
        
        if self.config['create_polynomial_features']:
            numerical_cols.extend([
                'tenure_squared', 'MonthlyCharges_squared',
                'tenure_log', 'TotalCharges_log'
            ])
        
        # Filter existing columns
        numerical_cols = [col for col in numerical_cols if col in df.columns]
        
        if fit:
            scaler = StandardScaler()
            df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
            self.scalers['standard'] = scaler
        else:
            if 'standard' in self.scalers:
                df[numerical_cols] = self.scalers['standard'].transform(df[numerical_cols])
        
        return df
    
    def prepare_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Complete feature preparation pipeline"""
        logger.info("Starting feature preparation...")
        
        # Handle missing values
        if self.config['handle_missing_values']:
            df = self.handle_missing_values(df)
        
        # Create advanced features
        df = self.create_advanced_features(df)
        
        # Encode categorical features
        df = self.encode_categorical_features(df, fit=fit)
        
        # Scale numerical features
        df = self.scale_numerical_features(df, fit=fit)
        
        # Select final features
        feature_cols = []
        
        # Add encoded categorical features
        for col in self.categorical_features:
            if f'{col}_encoded' in df.columns:
                feature_cols.append(f'{col}_encoded')
            elif col in df.columns and col == 'SeniorCitizen':
                feature_cols.append(col)
        
        # Add numerical features
        feature_cols.extend([
            'tenure', 'MonthlyCharges', 'TotalCharges',
            'AvgMonthlyCharges', 'ChargesPerService', 'NumServices',
            'IsMonthToMonth', 'HasLongTermContract', 'HasStreamingServices',
            'HasSecurityServices', 'IsElectronicPayment', 'IsAutoPayment'
        ])
        
        # Add interaction features
        if self.config['create_interaction_features']:
            feature_cols.extend([
                'SeniorWithDependents', 'MonthlyChargesXTenure', 'NoInternetXCharges'
            ])
        
        # Add polynomial features
        if self.config['create_polynomial_features']:
            feature_cols.extend([
                'tenure_squared', 'MonthlyCharges_squared',
                'tenure_log', 'TotalCharges_log'
            ])
        
        # Add tenure group
        if 'TenureGroup_encoded' in df.columns:
            feature_cols.append('TenureGroup_encoded')
        
        # Filter existing columns
        self.feature_names = [col for col in feature_cols if col in df.columns]
        
        logger.info(f"Prepared {len(self.feature_names)} features")
        
        return df
    
    def transform_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform target variable to binary"""
        df = df.copy()
        
        if self.config['target_column'] in df.columns:
            # Handle both numeric (0,1) and string ('Yes','No') target values
            target_col = df[self.config['target_column']]
            if target_col.dtype == 'object':
                df['target'] = (target_col == 'Yes').astype(int)
            else:
                df['target'] = target_col.astype(int)
        
        return df
    
    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data into train, validation, and test sets"""
        # Prepare target
        df = self.transform_target(df)
        
        # Get features and target
        X = df[self.feature_names]
        y = df['target']
        
        # First split: train+val and test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, 
            test_size=self.config['test_size'],
            random_state=self.config['random_state'],
            stratify=y
        )
        
        # Second split: train and validation
        val_size_adjusted = self.config['validation_size'] / (1 - self.config['test_size'])
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size_adjusted,
            random_state=self.config['random_state'],
            stratify=y_temp
        )
        
        # Combine features and target
        train_df = pd.concat([X_train, y_train], axis=1)
        val_df = pd.concat([X_val, y_val], axis=1)
        test_df = pd.concat([X_test, y_test], axis=1)
        
        logger.info(f"Data split - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        logger.info(f"Train churn rate: {y_train.mean():.2%}")
        logger.info(f"Val churn rate: {y_val.mean():.2%}")
        logger.info(f"Test churn rate: {y_test.mean():.2%}")
        
        return train_df, val_df, test_df
    
    def save_preprocessor(self, path: str):
        """Save preprocessing artifacts"""
        artifacts = {
            'encoders': self.encoders,
            'scalers': self.scalers,
            'feature_names': self.feature_names,
            'config': self.config
        }
        joblib.dump(artifacts, path)
        logger.info(f"Preprocessor saved to {path}")
    
    def load_preprocessor(self, path: str):
        """Load preprocessing artifacts"""
        artifacts = joblib.load(path)
        self.encoders = artifacts['encoders']
        self.scalers = artifacts['scalers']
        self.feature_names = artifacts['feature_names']
        self.config = artifacts['config']
        logger.info(f"Preprocessor loaded from {path}")


def preprocess_for_training(input_path: str, output_dir: str):
    """Preprocess data for SageMaker training"""
    # Initialize preprocessor
    preprocessor = TelcoFeatureEngineer()
    
    # Load data
    df = preprocessor.load_data(input_path)
    logger.info(f"Loaded data with shape: {df.shape}")
    
    # Prepare features
    df = preprocessor.prepare_features(df, fit=True)
    
    # Split data
    train_df, val_df, test_df = preprocessor.split_data(df)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save processed data
    train_df.to_csv(f"{output_dir}/train.csv", index=False, header=False)
    val_df.to_csv(f"{output_dir}/validation.csv", index=False, header=False)
    test_df.to_csv(f"{output_dir}/test.csv", index=False, header=True)
    
    # Save preprocessor
    preprocessor.save_preprocessor(f"{output_dir}/preprocessor.pkl")
    
    # Save feature names
    with open(f"{output_dir}/feature_names.json", 'w') as f:
        json.dump(preprocessor.feature_names, f)
    
    # Save metadata
    metadata = {
        'num_features': len(preprocessor.feature_names),
        'train_samples': len(train_df),
        'val_samples': len(val_df),
        'test_samples': len(test_df),
        'feature_names': preprocessor.feature_names,
        'target_column': 'target'
    }
    
    with open(f"{output_dir}/metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Preprocessing complete. Files saved to {output_dir}")
    
    return train_df, val_df, test_df


if __name__ == "__main__":
    # Example usage with XLSX file
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', type=str, 
                       default='data/Telco_customer_churn.xlsx',
                       help='Path to input data file')
    parser.add_argument('--output-dir', type=str,
                       default='data/processed',
                       help='Output directory for processed data')
    
    args = parser.parse_args()
    
    preprocess_for_training(args.input_path, args.output_dir)