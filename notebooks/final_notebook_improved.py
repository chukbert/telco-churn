# %% [markdown]
# # Telco Customer Churn: TensorFlow Showcase
# 
# This notebook implements and compares two models for customer churn prediction:
# 1. **Naive Bayes** - Simple probabilistic baseline model  
# 2. **TensorFlow DNN** - Advanced deep neural network with mixed precision
# 
# **Goal**: Demonstrate TensorFlow's superior performance over traditional machine learning methods
# 
# **Evaluation Metrics:**
# - ROC-AUC
# - Accuracy
# - F1-macro
# - Precision
# - Recall
# 
# **Advanced Features:**
# - Target-guided ordinal encoding for categorical variables
# - Advanced feature engineering with domain knowledge
# - SMOTE for handling class imbalance
# - Comprehensive model evaluation and comparison
# - Feature importance analysis

# %%
# Import required libraries
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Machine Learning
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, mixed_precision
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (roc_auc_score, accuracy_score, f1_score, 
                           classification_report, confusion_matrix, roc_curve,
                           precision_score, recall_score)
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Handle imbalanced data
from imblearn.over_sampling import SMOTE

# Add src to path
sys.path.append('../src')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('husl')

# %%
# GPU Detection and Setup
print("Detecting GPU...")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPU found: {gpus[0].name}")
    # Enable memory growth
    tf.config.experimental.set_memory_growth(gpus[0], True)
    # Enable mixed precision
    mixed_precision.set_global_policy('mixed_float16')
    print("Mixed precision enabled (FP16)")
else:
    print("No GPU found, using CPU")

# Check TensorFlow version
print(f"\nTensorFlow version: {tf.__version__}")
print(f"Scikit-learn version: Available")

# %% [markdown]
# ## 1. Data Loading and Advanced Preprocessing

# %%
# Load data
data_path = "data/Telco_customer_churn.xlsx"
df = pd.read_excel(data_path)
print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# Display first few rows to understand the data structure
print("\nFirst 5 rows:")
print(df.head())

print("\nDataset Info:")
df.info()

# Check all possible target columns
target_candidates = [col for col in df.columns if 'churn' in col.lower()]
print(f"\nPossible target columns: {target_candidates}")

# Check unique values in each target candidate
for col in target_candidates:
    print(f"\n{col} unique values: {df[col].unique()}")
    print(f"{col} value counts:\n{df[col].value_counts()}")

# Create proper target variable
if 'Churn Value' in df.columns:
    # Check if it's already numeric
    if df['Churn Value'].dtype in ['int64', 'float64']:
        print(f"Churn rate: {(df['Churn Value'] == 1).mean():.2%}")
    else:
        # Convert to numeric if it's text
        df['Churn Value'] = df['Churn Value'].map({'Yes': 1, 'No': 0})
        print(f"Churn rate: {(df['Churn Value'] == 1).mean():.2%}")
elif 'Churn Label' in df.columns:
    # Use Churn Label and convert it
    df['Churn Value'] = df['Churn Label'].map({'Yes': 1, 'No': 0})
    print(f"Churn rate: {(df['Churn Value'] == 1).mean():.2%}")
elif 'Churn' in df.columns:
    # Convert Yes/No to 1/0 if needed
    if df['Churn'].dtype == 'object':
        df['Churn Value'] = df['Churn'].map({'Yes': 1, 'No': 0})
    else:
        df['Churn Value'] = df['Churn']
    print(f"Churn rate: {(df['Churn Value'] == 1).mean():.2%}")
else:
    print("Warning: No suitable churn column found")
    print(f"Available columns: {list(df.columns)}")
    # Try to find any binary column that might be the target
    for col in df.columns:
        unique_vals = df[col].nunique()
        if unique_vals == 2:
            print(f"Binary column found: {col} with values {df[col].unique()}")

# Verify target variable
if 'Churn Value' in df.columns:
    print(f"\nTarget variable verification:")
    print(f"Churn Value distribution: {df['Churn Value'].value_counts().to_dict()}")
    print(f"Data type: {df['Churn Value'].dtype}")
    print(f"Any nulls: {df['Churn Value'].isnull().sum()}")
    
    # Additional debugging
    print(f"Unique values: {sorted(df['Churn Value'].unique())}")
    print(f"Min value: {df['Churn Value'].min()}")
    print(f"Max value: {df['Churn Value'].max()}")
    
    # Check if all values are the same
    if df['Churn Value'].nunique() == 1:
        print(f"\n⚠️  WARNING: All target values are the same: {df['Churn Value'].iloc[0]}")
        print(f"This means the dataset has no variation in the target variable.")
        print(f"Possible solutions:")
        print(f"1. Check if you're using the correct target column")
        print(f"2. Check if the data needs different preprocessing")
        print(f"3. Verify the original data source")
        
        # Show some examples of the original data
        print(f"\nSample of original data:")
        for col in df.columns:
            if 'churn' in col.lower():
                print(f"{col}: {df[col].head(10).tolist()}")
else:
    print("\n⚠️  No 'Churn Value' column created!")

# %%
# Data cleaning and preprocessing
print("Starting data preprocessing...")

# Drop customer ID if exists
id_columns = [col for col in df.columns if 'customer' in col.lower() or 'id' in col.lower()]
if id_columns:
    df = df.drop(id_columns, axis=1)
    print(f"Dropped ID columns: {id_columns}")

# Handle TotalCharges conversion
if 'Total Charges' in df.columns:
    df['Total Charges'] = pd.to_numeric(df['Total Charges'], errors='coerce')
elif 'TotalCharges' in df.columns:
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.rename(columns={'TotalCharges': 'Total Charges'})

# Handle tenure column naming
if 'tenure' in df.columns:
    df = df.rename(columns={'tenure': 'Tenure Months'})

# Handle monthly charges column naming
if 'MonthlyCharges' in df.columns:
    df = df.rename(columns={'MonthlyCharges': 'Monthly Charges'})

# Check for missing values
print(f"\nMissing values:")
print(df.isnull().sum())

# Handle missing values in Total Charges
if 'Total Charges' in df.columns and df['Total Charges'].isnull().sum() > 0:
    # Fill missing values with Monthly Charges for new customers (tenure = 0)
    df['Total Charges'].fillna(df['Monthly Charges'], inplace=True)
    print(f"Filled {df['Total Charges'].isnull().sum()} missing values in Total Charges")

# Handle missing values intelligently
initial_shape = df.shape[0]

# Don't drop rows just because Churn Reason is missing - that's expected for non-churners!
print(f"\\nMissing value analysis:")
missing_info = df.isnull().sum()
critical_missing = missing_info[missing_info > 0]
print(critical_missing)

# Only drop rows with missing values in critical columns (not Churn Reason)
critical_columns = [col for col in df.columns if col not in ['Churn Reason', 'Churn Score', 'CLTV']]
df_clean = df.dropna(subset=critical_columns)

print(f"\\nDropped {initial_shape - df_clean.shape[0]} rows with missing values in critical columns")
print(f"Kept data shape: {df_clean.shape}")

# Fill Churn Reason with 'No Churn' for non-churners
if 'Churn Reason' in df_clean.columns:
    df_clean['Churn Reason'] = df_clean['Churn Reason'].fillna('No Churn')
    print(f"Filled missing Churn Reason values with 'No Churn'")

# Update df to use the cleaned version
df = df_clean

# Identify feature types automatically
numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
if 'Churn Value' in numerical_features:
    numerical_features.remove('Churn Value')
if 'Churn' in numerical_features:
    numerical_features.remove('Churn')
    
categorical_features = df.select_dtypes(include=['object']).columns.tolist()
if 'Churn' in categorical_features:
    categorical_features.remove('Churn')

print(f"\nNumerical features ({len(numerical_features)}): {numerical_features}")
print(f"Categorical features ({len(categorical_features)}): {categorical_features}")
print(f"Final dataset shape: {df.shape}")

# %%
# Advanced Feature Engineering
def create_engineered_features(df):
    """Create advanced engineered features based on domain knowledge"""
    df = df.copy()
    
    # Ensure we have the right column names
    tenure_col = 'Tenure Months' if 'Tenure Months' in df.columns else 'tenure'
    monthly_col = 'Monthly Charges' if 'Monthly Charges' in df.columns else 'MonthlyCharges'
    total_col = 'Total Charges' if 'Total Charges' in df.columns else 'TotalCharges'
    
    # Revenue and usage features
    df['AvgMonthlyCharges'] = df[total_col] / (df[tenure_col] + 1)
    df['TotalChargesPerTenure'] = df[total_col] / (df[tenure_col] + 1)
    df['ChargesRatio'] = df[total_col] / (df[monthly_col] * (df[tenure_col] + 1))
    
    # Service counting features
    service_columns = []
    possible_services = ['Phone Service', 'PhoneService', 'Internet Service', 'InternetService',
                        'Streaming TV', 'StreamingTV', 'Streaming Movies', 'StreamingMovies',
                        'Online Security', 'OnlineSecurity', 'Online Backup', 'OnlineBackup',
                        'Device Protection', 'DeviceProtection', 'Tech Support', 'TechSupport']
    
    for service in possible_services:
        if service in df.columns:
            service_columns.append(service)
    
    if service_columns:
        # Count active services
        df['NumServices'] = df[service_columns].apply(
            lambda x: (x == 'Yes').sum() if x.dtype == 'object' else x.sum(), axis=1
        )
        
        # Streaming services
        streaming_cols = [col for col in service_columns if 'streaming' in col.lower()]
        if streaming_cols:
            df['HasStreamingServices'] = df[streaming_cols].apply(
                lambda x: (x == 'Yes').any() if x.dtype == 'object' else x.any(), axis=1
            ).astype(int)
        
        # Security services
        security_cols = [col for col in service_columns if any(word in col.lower() 
                        for word in ['security', 'backup', 'protection', 'support'])]
        if security_cols:
            df['HasSecurityServices'] = df[security_cols].apply(
                lambda x: (x == 'Yes').any() if x.dtype == 'object' else x.any(), axis=1
            ).astype(int)
    
    # Contract features
    if 'Contract' in df.columns:
        df['IsMonthToMonth'] = (df['Contract'] == 'Month-to-month').astype(int)
        df['HasLongTermContract'] = df['Contract'].isin(['One year', 'Two year']).astype(int)
    
    # Payment method features
    if 'Payment Method' in df.columns or 'PaymentMethod' in df.columns:
        payment_col = 'Payment Method' if 'Payment Method' in df.columns else 'PaymentMethod'
        df['IsElectronicPayment'] = df[payment_col].str.contains('electronic', case=False, na=False).astype(int)
        df['IsAutoPayment'] = df[payment_col].str.contains('automatic', case=False, na=False).astype(int)
        df['IsMailed'] = df[payment_col].str.contains('mail', case=False, na=False).astype(int)
    
    # Customer profile features
    if 'Partner' in df.columns and 'Dependents' in df.columns:
        df['HasFamily'] = ((df['Partner'] == 'Yes') | (df['Dependents'] == 'Yes')).astype(int)
        df['FamilySize'] = (df['Partner'] == 'Yes').astype(int) + (df['Dependents'] == 'Yes').astype(int)
    
    # Interaction features
    df['MonthlyChargesXTenure'] = df[monthly_col] * df[tenure_col]
    df['ChargesPerService'] = df[monthly_col] / (df.get('NumServices', 1) + 1)
    
    # Polynomial and log features for numerical variables
    df['TenureSquared'] = df[tenure_col] ** 2
    df['TenureCubed'] = df[tenure_col] ** 3
    df['MonthlyChargesSquared'] = df[monthly_col] ** 2
    df['TenureLog'] = np.log1p(df[tenure_col])
    df['TotalChargesLog'] = np.log1p(df[total_col])
    df['MonthlyChargesLog'] = np.log1p(df[monthly_col])
    
    # Binning features
    df['TenureGroup'] = pd.cut(df[tenure_col], bins=[0, 12, 24, 48, 100], 
                              labels=['New', 'Medium', 'Long', 'VeryLong'])
    df['ChargesGroup'] = pd.cut(df[monthly_col], bins=5, labels=['VeryLow', 'Low', 'Medium', 'High', 'VeryHigh'])
    
    return df

# Apply advanced feature engineering
print("Creating advanced engineered features...")
df_engineered = create_engineered_features(df)
print(f"Features engineered. New shape: {df_engineered.shape}")
print(f"New features added: {df_engineered.shape[1] - df.shape[1]}")

# %%
# Target-guided ordinal encoding for categorical variables
print("Applying target-guided ordinal encoding...")

def target_guided_encoding(df, target_col):
    """Apply target-guided ordinal encoding to categorical variables"""
    df_encoded = df.copy()
    encoding_mappings = {}
    
    # Get categorical columns
    categorical_cols = df_encoded.select_dtypes(include=['object']).columns.tolist()
    if target_col in categorical_cols:
        categorical_cols.remove(target_col)
    
    for col in categorical_cols:
        if col in df_encoded.columns:
            # Handle 'No internet service' and 'No phone service' standardization
            df_encoded[col] = df_encoded[col].replace({
                'No internet service': 'No',
                'No phone service': 'No'
            })
            
            # Calculate mean target value for each category
            target_means = df_encoded.groupby(col)[target_col].mean().sort_values()
            
            # Create ordinal mapping based on target means
            ordinal_mapping = {category: idx for idx, category in enumerate(target_means.index)}
            
            # Apply encoding
            df_encoded[col] = df_encoded[col].map(ordinal_mapping)
            encoding_mappings[col] = ordinal_mapping
            
            print(f"Encoded {col}: {len(ordinal_mapping)} categories")
    
    return df_encoded, encoding_mappings

# Apply target-guided encoding
df_encoded, encoding_mappings = target_guided_encoding(df_engineered, 'Churn Value')

# Prepare target variable
y = df_encoded['Churn Value'].values

# Drop non-feature columns
columns_to_drop = ['Churn Value', 'Churn Label', 'Churn Reason', 'Churn Score', 'CLTV', 
                   'Country', 'State', 'City', 'Lat Long', 'Zip Code', 'Latitude', 'Longitude',
                   'Churn']  # Add original Churn column if exists

# Also drop the new categorical binning features for now (will be encoded separately)
binning_features = ['TenureGroup', 'ChargesGroup']
columns_to_drop.extend(binning_features)

X = df_encoded.drop(columns_to_drop, axis=1, errors='ignore')

# Ensure all remaining columns are numeric
print("\nEnsuring all features are numeric...")
for col in X.columns:
    if X[col].dtype == 'object':
        print(f"Converting {col} to numeric")
        try:
            X[col] = pd.to_numeric(X[col], errors='coerce')
            X[col].fillna(X[col].median(), inplace=True)
        except:
            # Use label encoding as fallback
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))

# Remove any columns with all NaN values
X = X.dropna(axis=1, how='all')

print(f"\nFinal feature count: {X.shape[1]}")
print(f"Target distribution: {np.bincount(y)} (Class 0: {(y==0).sum()}, Class 1: {(y==1).sum()})")
print(f"Churn rate: {y.mean():.2%}")
print(f"\nSample of features: {list(X.columns)[:15]}")

# %%
# Handle class imbalance with SMOTE
print("Handling class imbalance with SMOTE...")
print(f"Original class distribution: {np.bincount(y)}")
print(f"Number of unique classes: {len(np.unique(y))}")
print(f"Class labels: {np.unique(y)}")

# Check if we have both classes before applying SMOTE
if len(np.unique(y)) < 2:
    print("\nERROR: Only one class found in target variable!")
    print("This usually means:")
    print("1. Wrong target column selected")
    print("2. Target column needs different encoding")
    print("3. Data filtering removed one class")
    print("\nPlease check the target variable creation above.")
    # Don't proceed with SMOTE
    X_balanced, y_balanced = X, y
else:
    # Apply SMOTE to balance the dataset
    oversample = SMOTE(random_state=42)
    X_balanced, y_balanced = oversample.fit_resample(X, y)
    
    print(f"Balanced class distribution: {np.bincount(y_balanced)}")
    print(f"Original dataset size: {X.shape[0]}")
    print(f"Balanced dataset size: {X_balanced.shape[0]}")

# Split the balanced data
# Use stratify only if we have both classes
stratify_param = y_balanced if len(np.unique(y_balanced)) > 1 else None

X_temp, X_test, y_temp, y_test = train_test_split(
    X_balanced, y_balanced, test_size=0.2, random_state=42, stratify=stratify_param
)

# Use stratify only if we have both classes in temp set
stratify_temp = y_temp if len(np.unique(y_temp)) > 1 else None

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.125, random_state=42, stratify=stratify_temp
)

print(f"\nData splits:")
print(f"Train: {X_train.shape} - Churn rate: {y_train.mean():.2%}")
print(f"Val:   {X_val.shape} - Churn rate: {y_val.mean():.2%}")
print(f"Test:  {X_test.shape} - Churn rate: {y_test.mean():.2%}")

# %%
# Feature scaling using StandardScaler
print("Scaling features...")
scaler = StandardScaler()

# Fit on train data and transform all sets
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print(f"Scaled data shapes:")
print(f"Train: {X_train_scaled.shape}")
print(f"Val:   {X_val_scaled.shape}")
print(f"Test:  {X_test_scaled.shape}")
print(f"Feature scaling complete")

# %% [markdown]
# ## 2. Model 1: Naive Bayes (Baseline)

# %%
# Naive Bayes - Simple baseline model
print("Training Naive Bayes baseline model...")

# Check if we have both classes before training
if len(np.unique(y_train)) < 2:
    print("\n❌ ERROR: Cannot train models with only one class in training data!")
    print(f"Training set has only class: {np.unique(y_train)}")
    print("Please fix the target variable issue before proceeding.")
    print("\nSkipping model training...")
else:
    print(f"✅ Training set has {len(np.unique(y_train))} classes: {np.unique(y_train)}")

# Create and train Naive Bayes model (simple, no hyperparameter tuning needed)
nb_model = GaussianNB()

# Fit the model (very fast training)
print("Training Naive Bayes model...")
nb_model.fit(X_train_scaled, y_train)

print("Naive Bayes training completed!")

# %%
# Evaluate Naive Bayes
print("Evaluating Naive Bayes baseline...")

nb_pred_val = nb_model.predict_proba(X_val_scaled)[:, 1]
nb_pred_test = nb_model.predict_proba(X_test_scaled)[:, 1]

# Binary predictions
nb_pred_val_binary = nb_model.predict(X_val_scaled)
nb_pred_test_binary = nb_model.predict(X_test_scaled)

# Calculate comprehensive metrics
nb_val_auc = roc_auc_score(y_val, nb_pred_val)
nb_test_auc = roc_auc_score(y_test, nb_pred_test)

nb_val_accuracy = accuracy_score(y_val, nb_pred_val_binary)
nb_test_accuracy = accuracy_score(y_test, nb_pred_test_binary)

nb_val_f1 = f1_score(y_val, nb_pred_val_binary, average='macro')
nb_test_f1 = f1_score(y_test, nb_pred_test_binary, average='macro')

nb_val_precision = precision_score(y_val, nb_pred_val_binary, average='macro')
nb_test_precision = precision_score(y_test, nb_pred_test_binary, average='macro')

nb_val_recall = recall_score(y_val, nb_pred_val_binary, average='macro')
nb_test_recall = recall_score(y_test, nb_pred_test_binary, average='macro')

print(f"\nNaive Bayes Results:")
print(f"Validation - AUC: {nb_val_auc:.4f}, Accuracy: {nb_val_accuracy:.4f}, F1: {nb_val_f1:.4f}")
print(f"           - Precision: {nb_val_precision:.4f}, Recall: {nb_val_recall:.4f}")
print(f"Test       - AUC: {nb_test_auc:.4f}, Accuracy: {nb_test_accuracy:.4f}, F1: {nb_test_f1:.4f}")
print(f"           - Precision: {nb_test_precision:.4f}, Recall: {nb_test_recall:.4f}")

# %% [markdown]
# ## 3. Model 2: TensorFlow DNN (Advanced)

# %%
# Build TensorFlow DNN Model
print("Building TensorFlow DNN model...")

def build_tensorflow_dnn(input_dim, dropout_rate=0.3):
    """
    Build a deep neural network for churn prediction.
    """
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

# Create and compile the TensorFlow DNN model
tf_model = build_tensorflow_dnn(X_train_scaled.shape[1])

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

print(f"TensorFlow DNN parameters: {tf_model.count_params():,}")
tf_model.summary()

# %%
# Train TensorFlow DNN
print("Training TensorFlow DNN...")

# Calculate class weights
neg_weight = len(y_train) / (2 * np.sum(y_train == 0))
pos_weight = len(y_train) / (2 * np.sum(y_train == 1))
class_weights = {0: neg_weight, 1: pos_weight}

# Train the model
history = tf_model.fit(
    X_train_scaled, y_train,
    validation_data=(X_val_scaled, y_val),
    epochs=20,  # Reduced for faster training
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

# %%
# Evaluate TensorFlow DNN
print("Evaluating TensorFlow DNN...")

tf_pred_val = tf_model.predict(X_val_scaled, verbose=0)[:, 0]
tf_pred_test = tf_model.predict(X_test_scaled, verbose=0)[:, 0]

# Binary predictions
tf_pred_val_binary = (tf_pred_val > 0.5).astype(int)
tf_pred_test_binary = (tf_pred_test > 0.5).astype(int)

# Calculate comprehensive metrics
tf_val_auc = roc_auc_score(y_val, tf_pred_val)
tf_test_auc = roc_auc_score(y_test, tf_pred_test)

tf_val_accuracy = accuracy_score(y_val, tf_pred_val_binary)
tf_test_accuracy = accuracy_score(y_test, tf_pred_test_binary)

tf_val_f1 = f1_score(y_val, tf_pred_val_binary, average='macro')
tf_test_f1 = f1_score(y_test, tf_pred_test_binary, average='macro')

tf_val_precision = precision_score(y_val, tf_pred_val_binary, average='macro')
tf_test_precision = precision_score(y_test, tf_pred_test_binary, average='macro')

tf_val_recall = recall_score(y_val, tf_pred_val_binary, average='macro')
tf_test_recall = recall_score(y_test, tf_pred_test_binary, average='macro')

print(f"\nTensorFlow DNN Results:")
print(f"Validation - AUC: {tf_val_auc:.4f}, Accuracy: {tf_val_accuracy:.4f}, F1: {tf_val_f1:.4f}")
print(f"           - Precision: {tf_val_precision:.4f}, Recall: {tf_val_recall:.4f}")
print(f"Test       - AUC: {tf_test_auc:.4f}, Accuracy: {tf_test_accuracy:.4f}, F1: {tf_test_f1:.4f}")
print(f"           - Precision: {tf_test_precision:.4f}, Recall: {tf_test_recall:.4f}")

# %% [markdown]
# ## 3. Comprehensive Two-Model Comparison

# %%
# Two-model comparison results
# Create comprehensive results dataframe
results_df = pd.DataFrame({
    'Model': ['Naive Bayes', 'TensorFlow DNN'],
    'Val_AUC': [nb_val_auc, tf_val_auc],
    'Test_AUC': [nb_test_auc, tf_test_auc],
    'Val_Accuracy': [nb_val_accuracy, tf_val_accuracy],
    'Test_Accuracy': [nb_test_accuracy, tf_test_accuracy],
    'Val_F1_Macro': [nb_val_f1, tf_val_f1],
    'Test_F1_Macro': [nb_test_f1, tf_test_f1],
    'Val_Precision': [nb_val_precision, tf_val_precision],
    'Test_Precision': [nb_test_precision, tf_test_precision],
    'Val_Recall': [nb_val_recall, tf_val_recall],
    'Test_Recall': [nb_test_recall, tf_test_recall],
    'Parameters': [
        f"{X_train_scaled.shape[1]} features (no tuning)",
        f"{tf_model.count_params():,} parameters"
    ],
    'Architecture': [
        'Probabilistic (Gaussian)',
        'Deep Neural Network'
    ]
})

print("Final Results Summary (Two Models):")
print("=" * 120)

# Only show results for models that have been run
for _, row in results_df.iterrows():
    if row['Test_AUC'] > 0:  # Only show if model was actually run
        print(f"\n{row['Model']}:")
        print(f"  Architecture: {row['Architecture']}")
        print(f"  Parameters: {row['Parameters']}")
        print(f"  Validation - AUC: {row['Val_AUC']:.4f}, Accuracy: {row['Val_Accuracy']:.4f}, F1: {row['Val_F1_Macro']:.4f}")
        print(f"             - Precision: {row['Val_Precision']:.4f}, Recall: {row['Val_Recall']:.4f}")
        print(f"  Test       - AUC: {row['Test_AUC']:.4f}, Accuracy: {row['Test_Accuracy']:.4f}, F1: {row['Test_F1_Macro']:.4f}")
        print(f"             - Precision: {row['Test_Precision']:.4f}, Recall: {row['Test_Recall']:.4f}")

print("\n" + "=" * 120)

# Model rankings by each metric
print("\nModel Rankings by Metric (Test Set):")
for metric in ['Test_AUC', 'Test_Accuracy', 'Test_F1_Macro', 'Test_Precision', 'Test_Recall']:
    metric_name = metric.replace('Test_', '').replace('_', '-')
    ranked = results_df.sort_values(metric, ascending=False)
    print(f"\n{metric_name}:")
    for i, (_, row) in enumerate(ranked.iterrows(), 1):
        print(f"  {i}. {row['Model']}: {row[metric]:.4f}")

# Best overall model
print("\nOverall Performance Analysis:")
# Include all metrics in the average
avg_scores = results_df[['Test_AUC', 'Test_Accuracy', 'Test_F1_Macro', 'Test_Precision', 'Test_Recall']].mean(axis=1)
best_idx = avg_scores.idxmax()
best_overall = results_df.loc[best_idx]

print(f"Best overall model: {best_overall['Model']}")
print(f"Average test score: {avg_scores.loc[best_idx]:.4f}")
print(f"Architecture: {best_overall['Architecture']}")

# Performance comparison
nb_avg = avg_scores.iloc[0]
tf_avg = avg_scores.iloc[1]

if tf_avg > nb_avg:
    improvement = ((tf_avg - nb_avg) / nb_avg) * 100
    print(f"TensorFlow DNN outperforms Naive Bayes by {improvement:.1f}%")
elif nb_avg > tf_avg:
    improvement = ((nb_avg - tf_avg) / tf_avg) * 100
    print(f"Naive Bayes outperforms TensorFlow DNN by {improvement:.1f}%")
else:
    print("Both models perform similarly")

# Performance insights
print(f"\nPerformance Insights:")
nb_results = results_df[results_df['Model'] == 'Naive Bayes'].iloc[0]
tf_results = results_df[results_df['Model'] == 'TensorFlow DNN'].iloc[0]

print(f"Model Complexity:")
print(f"  - Naive Bayes: Simple probabilistic model with {nb_results['Parameters']}")
print(f"  - TensorFlow DNN: Complex neural network with {tf_results['Parameters']}")

print(f"\nKey Findings:")
if tf_results['Test_AUC'] > nb_results['Test_AUC']:
    auc_improvement = ((tf_results['Test_AUC'] - nb_results['Test_AUC']) / nb_results['Test_AUC']) * 100
    print(f"  - TensorFlow DNN shows {auc_improvement:.1f}% better AUC than Naive Bayes")

if tf_results['Test_F1_Macro'] > nb_results['Test_F1_Macro']:
    f1_improvement = ((tf_results['Test_F1_Macro'] - nb_results['Test_F1_Macro']) / nb_results['Test_F1_Macro']) * 100
    print(f"  - TensorFlow DNN shows {f1_improvement:.1f}% better F1-score than Naive Bayes")
    
# Recommend based on performance vs complexity trade-off
if tf_results['Test_AUC'] > nb_results['Test_AUC'] + 0.05:  # 5% improvement threshold for simple baseline
    print(f"  - Recommendation: TensorFlow DNN (significant performance improvement over baseline)")
elif abs(tf_results['Test_AUC'] - nb_results['Test_AUC']) < 0.02:  # Within 2%
    print(f"  - Recommendation: Consider simpler model if performance gap is minimal")
else:
    print(f"  - Recommendation: TensorFlow DNN justifies complexity with better performance")

print("\nTo see complete comparison, run all model training cells in order.")

# %% [markdown]
# ## 4. Visualization and Analysis

# %%
# Plot ROC curves
plt.figure(figsize=(10, 8))

# Calculate ROC curves
models = {
    'Naive Bayes': nb_pred_test,
    'TensorFlow DNN': tf_pred_test
}

colors = ['#1f77b4', '#ff7f0e']

for (name, predictions), color in zip(models.items(), colors):
    fpr, tpr, _ = roc_curve(y_test, predictions)
    auc = roc_auc_score(y_test, predictions)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.4f})', 
             linewidth=3, color=color)

plt.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curves - Two-Model Comparison', fontsize=14, fontweight='bold')
plt.legend(loc="lower right", fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %%
# Performance metrics comparison
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

metrics = ['Test_AUC', 'Test_Accuracy', 'Test_F1_Macro', 'Test_Precision', 'Test_Recall']
metric_names = ['ROC-AUC', 'Accuracy', 'F1-macro', 'Precision', 'Recall']
colors = ['skyblue', 'lightcoral']

# Flatten axes for easier indexing
axes_flat = axes.flatten()

for i, (metric, name) in enumerate(zip(metrics, metric_names)):
    if i < len(axes_flat):
        values = results_df[metric].values
        models = results_df['Model'].values
        
        bars = axes_flat[i].bar(models, values, color=colors)
        axes_flat[i].set_ylabel(name, fontsize=12)
        axes_flat[i].set_title(f'{name} Comparison', fontsize=14, fontweight='bold')
        axes_flat[i].grid(True, alpha=0.3, axis='y')
        axes_flat[i].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            axes_flat[i].annotate(f'{value:.4f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom',
                            fontsize=11, fontweight='bold')
        
        # Set y-axis limits for better visualization
        y_min = max(0, min(values) - 0.02)
        y_max = min(1.0, max(values) + 0.02)
        axes_flat[i].set_ylim(y_min, y_max)

# Hide the last subplot if we have an odd number of metrics
if len(metrics) < len(axes_flat):
    axes_flat[-1].set_visible(False)

plt.tight_layout()
plt.show()

# %%
# Training history for TensorFlow DNN
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Loss
axes[0, 0].plot(history.history['loss'], label='Train Loss')
axes[0, 0].plot(history.history['val_loss'], label='Val Loss')
axes[0, 0].set_title('Training Loss', fontsize=12)
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# AUC
axes[0, 1].plot(history.history['auc'], label='Train AUC')
axes[0, 1].plot(history.history['val_auc'], label='Val AUC')
axes[0, 1].set_title('Training AUC', fontsize=12)
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('AUC')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Accuracy
axes[1, 0].plot(history.history['accuracy'], label='Train Accuracy')
axes[1, 0].plot(history.history['val_accuracy'], label='Val Accuracy')
axes[1, 0].set_title('Training Accuracy', fontsize=12)
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Accuracy')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Precision and Recall
axes[1, 1].plot(history.history['precision'], label='Train Precision')
axes[1, 1].plot(history.history['val_precision'], label='Val Precision')
axes[1, 1].plot(history.history['recall'], label='Train Recall')
axes[1, 1].plot(history.history['val_recall'], label='Val Recall')
axes[1, 1].set_title('Precision & Recall', fontsize=12)
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Score')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.suptitle('TensorFlow DNN Training History', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# %%
# Feature analysis using correlation with target
plt.figure(figsize=(12, 8))

# Calculate correlation between features and target
feature_correlations = pd.DataFrame({
    'Feature': X.columns,
    'Correlation': [abs(np.corrcoef(X[col], y)[0, 1]) for col in X.columns]
}).sort_values('Correlation', ascending=True)

# Get top 20 features by correlation
top_features = feature_correlations.tail(20)

plt.barh(range(len(top_features)), top_features['Correlation'])
plt.yticks(range(len(top_features)), top_features['Feature'])
plt.xlabel('Absolute Correlation with Churn')
plt.title('Top 20 Feature Correlations with Churn', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.show()

# %%
# Confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Naive Bayes confusion matrix
cm_nb = confusion_matrix(y_test, nb_pred_test_binary)
sns.heatmap(cm_nb, annot=True, fmt='d', cmap='Blues', ax=axes[0], 
            xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
axes[0].set_title('Naive Bayes\nConfusion Matrix', fontsize=12)
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')

# TensorFlow DNN confusion matrix
cm_tf = confusion_matrix(y_test, tf_pred_test_binary)
sns.heatmap(cm_tf, annot=True, fmt='d', cmap='Oranges', ax=axes[1], 
            xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
axes[1].set_title('TensorFlow DNN\nConfusion Matrix', fontsize=12)
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')

plt.tight_layout()
plt.show()

# Display detailed classification reports
print("\nDetailed Classification Metrics:")
print("=" * 80)

models_metrics = [
    ("Naive Bayes", y_test, nb_pred_test_binary),
    ("TensorFlow DNN", y_test, tf_pred_test_binary)
]

for model_name, y_true, y_pred in models_metrics:
    print(f"\n{model_name}:")
    print(classification_report(y_true, y_pred, target_names=['No Churn', 'Churn']))

# %% [markdown]
# ## 5. Key Insights and Final Recommendations

# %%
# Generate comprehensive insights
print("\nKey Insights and Analysis:")
print("=" * 80)

nb_row = results_df[results_df['Model'] == 'Naive Bayes'].iloc[0]
tf_row = results_df[results_df['Model'] == 'TensorFlow DNN'].iloc[0]

# Model performance comparison
print("\n1. Model Performance Comparison:")
metrics_to_compare = ['Test_AUC', 'Test_Accuracy', 'Test_F1_Macro', 'Test_Precision', 'Test_Recall']

for metric in metrics_to_compare:
    nb_score = nb_row[metric]
    tf_score = tf_row[metric]
    
    if tf_score > nb_score:
        improvement = ((tf_score - nb_score) / nb_score) * 100
        winner = "TensorFlow DNN"
        print(f"   - {metric.replace('Test_', '').replace('_', '-')}: {winner} wins ({tf_score:.4f} vs {nb_score:.4f}, +{improvement:.1f}%)")
    elif nb_score > tf_score:
        improvement = ((nb_score - tf_score) / tf_score) * 100
        winner = "Naive Bayes"
        print(f"   - {metric.replace('Test_', '').replace('_', '-')}: {winner} wins ({nb_score:.4f} vs {tf_score:.4f}, +{improvement:.1f}%)")
    else:
        print(f"   - {metric.replace('Test_', '').replace('_', '-')}: Tie ({nb_score:.4f})")

# Stability analysis
print("\n2. Model Stability Analysis (Validation vs Test):")
for _, row in results_df.iterrows():
    auc_gap = abs(row['Val_AUC'] - row['Test_AUC'])
    acc_gap = abs(row['Val_Accuracy'] - row['Test_Accuracy'])
    f1_gap = abs(row['Val_F1_Macro'] - row['Test_F1_Macro'])
    avg_gap = (auc_gap + acc_gap + f1_gap) / 3
    
    stability = "Excellent" if avg_gap < 0.01 else "Good" if avg_gap < 0.02 else "Fair" if avg_gap < 0.05 else "Poor"
    print(f"   - {row['Model']}: {stability} stability (avg gap: {avg_gap:.4f})")

# Model complexity analysis
print("\n3. Model Complexity Analysis:")
print(f"   - Naive Bayes: Simple probabilistic model, assumes feature independence")
print(f"   - TensorFlow DNN: Deep neural network with {tf_row['Parameters']}")
print(f"   - Complexity trade-off: DNN is significantly more complex than Naive Bayes")

# Data preprocessing insights
print("\n4. Data Preprocessing Impact:")
print(f"   - Applied SMOTE for class balancing: {X_balanced.shape[0]:,} samples (from {X.shape[0]:,})")
print(f"   - Target-guided ordinal encoding for categorical features")
print(f"   - Advanced feature engineering: {X.shape[1]} total features")
print(f"   - StandardScaler normalization for numerical stability")

# Final recommendation with reasoning
print("\n5. Production Recommendation:")

# Calculate weighted average (AUC gets more weight for churn prediction)
nb_weighted = (nb_row['Test_AUC'] * 0.4 + nb_row['Test_F1_Macro'] * 0.3 + 
               nb_row['Test_Precision'] * 0.15 + nb_row['Test_Recall'] * 0.15)
tf_weighted = (tf_row['Test_AUC'] * 0.4 + tf_row['Test_F1_Macro'] * 0.3 + 
               tf_row['Test_Precision'] * 0.15 + tf_row['Test_Recall'] * 0.15)

performance_diff = abs(tf_weighted - nb_weighted)

if tf_weighted > nb_weighted + 0.02:  # 2% threshold
    print(f"   [RECOMMENDED] TensorFlow DNN for production")
    print(f"   - Weighted performance: {tf_weighted:.4f} vs {nb_weighted:.4f}")
    print(f"   - Justification: Significant performance improvement ({((tf_weighted - nb_weighted) / nb_weighted * 100):.1f}%)")
    print(f"   - Best for: High-stakes churn prediction where accuracy is paramount")
elif nb_weighted > tf_weighted + 0.02:
    print(f"   [RECOMMENDED] Naive Bayes for production")
    print(f"   - Weighted performance: {nb_weighted:.4f} vs {tf_weighted:.4f}")
    print(f"   - Justification: Better performance with much lower complexity")
elif performance_diff < 0.01:  # Very close performance
    print(f"   [RECOMMENDED] Naive Bayes for production")
    print(f"   - Weighted performance: Similar ({nb_weighted:.4f} vs {tf_weighted:.4f})")
    print(f"   - Justification: Comparable performance, much simpler and faster")
else:
    print(f"   [CONTEXT-DEPENDENT] Both models viable")
    print(f"   - Performance difference: {performance_diff:.4f}")
    print(f"   - Choose based on: interpretability needs, computational resources, deployment constraints")

print("\n6. Use Case Recommendations:")
print(f"   - Real-time scoring: Naive Bayes (very fast inference)")
print(f"   - Batch processing: Either model suitable")
print(f"   - Interpretability required: Naive Bayes (probabilistic interpretation)")
print(f"   - Maximum performance: TensorFlow DNN (if performance gap significant)")
print(f"   - Limited computational resources: Naive Bayes")

# Feature correlation analysis for insights
print("\n7. Feature Analysis (Correlation with Churn):")

# Calculate feature correlations with target
feature_correlations = pd.DataFrame({
    'Feature': X.columns,
    'Correlation': [np.corrcoef(X[col], y)[0, 1] for col in X.columns]
}).sort_values('Correlation', key=abs, ascending=False)

print("\nTop 10 Most Correlated Features:")
for i, (_, row) in enumerate(feature_correlations.head(10).iterrows(), 1):
    direction = "positive" if row['Correlation'] > 0 else "negative"
    print(f"   {i:2d}. {row['Feature']:<25} ({row['Correlation']:.4f}) - {direction} correlation")

print("\nTop 5 Positive Correlations (Increase Churn):")
positive_corr = feature_correlations[feature_correlations['Correlation'] > 0].head(5)
for i, (_, row) in enumerate(positive_corr.iterrows(), 1):
    print(f"   {i}. {row['Feature']:<25} (+{row['Correlation']:.4f})")

print("\nTop 5 Negative Correlations (Decrease Churn):")
negative_corr = feature_correlations[feature_correlations['Correlation'] < 0].head(5)
for i, (_, row) in enumerate(negative_corr.iterrows(), 1):
    print(f"   {i}. {row['Feature']:<25} ({row['Correlation']:.4f})")

# Final results summary for saving
final_results_df = pd.DataFrame({
    'Model': ['Naive Bayes', 'TensorFlow DNN'],
    'Val_AUC': [nb_val_auc, tf_val_auc],
    'Test_AUC': [nb_test_auc, tf_test_auc],
    'Val_Accuracy': [nb_val_accuracy, tf_val_accuracy],
    'Test_Accuracy': [nb_test_accuracy, tf_test_accuracy],
    'Val_F1_Macro': [nb_val_f1, tf_val_f1],
    'Test_F1_Macro': [nb_test_f1, tf_test_f1],
    'Val_Precision': [nb_val_precision, tf_val_precision],
    'Test_Precision': [nb_test_precision, tf_test_precision],
    'Val_Recall': [nb_val_recall, tf_val_recall],
    'Test_Recall': [nb_test_recall, tf_test_recall],
    'Parameters': [
        f"{X_train_scaled.shape[1]} features (no tuning)",
        f"{tf_model.count_params():,} parameters"
    ]
})

# Determine best model for each metric
print("\nBest Model by Metric (Test Set):")
best_auc = final_results_df.loc[final_results_df['Test_AUC'].idxmax(), 'Model']
best_acc = final_results_df.loc[final_results_df['Test_Accuracy'].idxmax(), 'Model']
best_f1 = final_results_df.loc[final_results_df['Test_F1_Macro'].idxmax(), 'Model']
best_precision = final_results_df.loc[final_results_df['Test_Precision'].idxmax(), 'Model']
best_recall = final_results_df.loc[final_results_df['Test_Recall'].idxmax(), 'Model']

print(f"  ROC-AUC: {best_auc} ({final_results_df['Test_AUC'].max():.4f})")
print(f"  Accuracy: {best_acc} ({final_results_df['Test_Accuracy'].max():.4f})")
print(f"  F1-macro: {best_f1} ({final_results_df['Test_F1_Macro'].max():.4f})")
print(f"  Precision: {best_precision} ({final_results_df['Test_Precision'].max():.4f})")
print(f"  Recall: {best_recall} ({final_results_df['Test_Recall'].max():.4f})")

# Save results to CSV
filename = 'telco_churn_two_model_results.csv'
final_results_df.to_csv(filename, index=False)
print(f"\nFinal Report saved to: {filename}")
print(f"Results include comprehensive metrics for {len(final_results_df)} models")

print("\n" + "=" * 80)
print("Analysis Complete! 📊")
print("\nNext Steps:")
print("1. Review feature importance for business insights")
print("2. Consider A/B testing the recommended model")
print("3. Monitor model performance in production")
print("4. Retrain periodically with new data")