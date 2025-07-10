"""
TensorFlow DNN with distributed training and mixed precision for SageMaker
Implements data parallel training with automatic mixed precision (AMP)
Updated for 33 features from preprocessing
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os
import json
import argparse
import logging
from typing import Dict, Tuple, Optional

# Enable mixed precision training
from tensorflow.keras.mixed_precision import Policy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TelcoChurnDNN:
    """TensorFlow DNN for customer churn prediction with DDP and mixed precision"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.model = None
        self.strategy = None
        
        # Set up distributed strategy
        self._setup_distributed_strategy()
        
        # Enable mixed precision if specified
        if config.get('use_mixed_precision', True):
            self._enable_mixed_precision()
    
    def _setup_distributed_strategy(self):
        """Set up distributed training strategy"""
        # Check if running in distributed mode
        if len(tf.config.list_physical_devices('GPU')) > 1:
            logger.info("Using MirroredStrategy for multi-GPU training")
            self.strategy = tf.distribute.MirroredStrategy()
        else:
            logger.info("Using default strategy (single GPU or CPU)")
            self.strategy = tf.distribute.get_strategy()
        
        logger.info(f"Number of devices: {self.strategy.num_replicas_in_sync}")
    
    def _enable_mixed_precision(self):
        """Enable automatic mixed precision training"""
        policy = Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        logger.info(f"Mixed precision policy: {policy.name}")
        logger.info(f"Compute dtype: {policy.compute_dtype}")
        logger.info(f"Variable dtype: {policy.variable_dtype}")
    
    def build_model(self, input_dim: int) -> keras.Model:
        """Build DNN model with advanced architecture"""
        with self.strategy.scope():
            # Input layer
            inputs = layers.Input(shape=(input_dim,), name='features')
            
            # Initial batch normalization
            x = layers.BatchNormalization()(inputs)
            
            # Deep architecture with residual connections
            hidden_units = self.config.get('hidden_units', [512, 256, 128, 64])
            dropout_rate = self.config.get('dropout_rate', 0.3)
            
            # Store intermediate outputs for skip connections
            skip_outputs = []
            
            for i, units in enumerate(hidden_units):
                # Dense layer
                x = layers.Dense(
                    units,
                    activation=None,  # Apply activation after batch norm
                    kernel_initializer='he_normal',
                    name=f'dense_{i}'
                )(x)
                
                # Batch normalization
                x = layers.BatchNormalization(name=f'bn_{i}')(x)
                
                # Activation
                x = layers.Activation('relu', name=f'relu_{i}')(x)
                
                # Dropout
                x = layers.Dropout(dropout_rate, name=f'dropout_{i}')(x)
                
                # Add skip connection every 2 layers
                if i > 0 and i % 2 == 0 and i < len(hidden_units) - 1:
                    if skip_outputs:
                        # Ensure dimensions match for skip connection
                        skip = skip_outputs[-1]
                        if skip.shape[-1] == x.shape[-1]:
                            x = layers.Add(name=f'skip_{i}')([x, skip])
                
                skip_outputs.append(x)
            
            # Output layer
            outputs = layers.Dense(1, activation='sigmoid', name='output')(x)
            
            # For mixed precision, we need to ensure output is float32
            if tf.keras.mixed_precision.global_policy().name == 'mixed_float16':
                outputs = layers.Activation('linear', dtype='float32', name='output_float32')(outputs)
            
            # Create model
            model = keras.Model(inputs=inputs, outputs=outputs, name='telco_churn_dnn')
            
            # Compile with optimizer that works well with mixed precision
            optimizer = self._get_optimizer()
            
            model.compile(
                optimizer=optimizer,
                loss='binary_crossentropy',
                metrics=[
                    'accuracy',
                    keras.metrics.AUC(name='auc'),
                    keras.metrics.Precision(name='precision'),
                    keras.metrics.Recall(name='recall')
                ]
            )
            
            self.model = model
            logger.info(f"Model built successfully with {model.count_params()} parameters")
            
            return model
    
    def _get_optimizer(self):
        """Get optimizer with loss scaling for mixed precision"""
        learning_rate = self.config.get('learning_rate', 0.001)
        
        # Create base optimizer
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        
        # Wrap with loss scale optimizer for mixed precision
        if tf.keras.mixed_precision.global_policy().name == 'mixed_float16':
            optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
        
        return optimizer
    
    def create_dataset(self, file_path: str, batch_size: int, 
                      is_training: bool = True) -> tf.data.Dataset:
        """Create optimized tf.data.Dataset for training"""
        # Define feature columns (33 features based on preprocessing)
        feature_columns = list(range(self.config.get('num_features', 33)))
        label_column = self.config.get('num_features', 33)
        
        def parse_csv(line):
            # Parse CSV line
            defaults = [[0.0]] * (len(feature_columns) + 1)
            parsed = tf.io.decode_csv(line, defaults)
            features = tf.stack(parsed[:-1])
            label = parsed[-1]
            return features, label
        
        # Create dataset
        dataset = tf.data.TextLineDataset(file_path)
        dataset = dataset.map(parse_csv, num_parallel_calls=tf.data.AUTOTUNE)
        
        if is_training:
            dataset = dataset.shuffle(buffer_size=10000)
            dataset = dataset.repeat()
        
        # Batch and prefetch
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def train(self, train_path: str, val_path: str, 
              epochs: int, batch_size: int, steps_per_epoch: int = None):
        """Train the model with distributed strategy"""
        # Calculate global batch size
        global_batch_size = batch_size * self.strategy.num_replicas_in_sync
        
        # Create datasets
        train_dataset = self.create_dataset(train_path, global_batch_size, is_training=True)
        val_dataset = self.create_dataset(val_path, global_batch_size, is_training=False)
        
        # Callbacks
        callbacks = self._get_callbacks()
        
        # Calculate steps if not provided
        if steps_per_epoch is None:
            # Estimate based on file size
            steps_per_epoch = 5000 // global_batch_size
        
        validation_steps = 1000 // global_batch_size
        
        # Train model
        history = self.model.fit(
            train_dataset,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            validation_data=val_dataset,
            validation_steps=validation_steps,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def _get_callbacks(self):
        """Get training callbacks"""
        callbacks = []
        
        # Model checkpoint
        checkpoint_callback = keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(self.config.get('output_dir', '/opt/ml/model'), 'best_model.h5'),
            monitor='val_auc',
            mode='max',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        )
        callbacks.append(checkpoint_callback)
        
        # Early stopping
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_auc',
            patience=5,
            mode='max',
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stopping)
        
        # Reduce learning rate on plateau
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        )
        callbacks.append(reduce_lr)
        
        # Custom callback for mixed precision monitoring
        if tf.keras.mixed_precision.global_policy().name == 'mixed_float16':
            class MixedPrecisionLogger(keras.callbacks.Callback):
                def on_epoch_end(self, epoch, logs=None):
                    if hasattr(self.model.optimizer, 'loss_scale'):
                        loss_scale = self.model.optimizer.loss_scale()
                        logger.info(f"Loss scale: {loss_scale}")
            
            callbacks.append(MixedPrecisionLogger())
        
        return callbacks
    
    def save_model(self, output_dir: str):
        """Save model artifacts for SageMaker"""
        # Save in TensorFlow SavedModel format
        tf_model_path = os.path.join(output_dir, 'tensorflow_model/1')
        self.model.save(tf_model_path, save_format='tf')
        logger.info(f"Model saved to {tf_model_path}")
        
        # Save model architecture
        model_json = self.model.to_json()
        with open(os.path.join(output_dir, 'model_architecture.json'), 'w') as f:
            f.write(model_json)
        
        # Save training config
        with open(os.path.join(output_dir, 'training_config.json'), 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def evaluate(self, test_path: str, batch_size: int) -> Dict:
        """Evaluate model performance"""
        test_dataset = self.create_dataset(test_path, batch_size, is_training=False)
        
        # Get number of test samples (approximate)
        test_steps = 1000 // batch_size
        
        results = self.model.evaluate(test_dataset, steps=test_steps, verbose=1)
        
        metrics = dict(zip(self.model.metrics_names, results))
        logger.info(f"Test metrics: {metrics}")
        
        return metrics


def main():
    """SageMaker training script entry point"""
    parser = argparse.ArgumentParser()
    
    # SageMaker specific arguments
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR', '/opt/ml/model'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train'))
    parser.add_argument('--validation', type=str, default=os.environ.get('SM_CHANNEL_VALIDATION', '/opt/ml/input/data/validation'))
    parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST', '/opt/ml/input/data/test'))
    
    # Hyperparameters
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--hidden-units', type=str, default='512,256,128,64')
    parser.add_argument('--dropout-rate', type=float, default=0.3)
    parser.add_argument('--num-features', type=int, default=33)  # Updated for actual dataset
    parser.add_argument('--use-mixed-precision', type=bool, default=True)
    
    args = parser.parse_args()
    
    # Parse hidden units
    hidden_units = [int(x) for x in args.hidden_units.split(',')]
    
    # Configuration
    config = {
        'learning_rate': args.learning_rate,
        'hidden_units': hidden_units,
        'dropout_rate': args.dropout_rate,
        'num_features': args.num_features,
        'use_mixed_precision': args.use_mixed_precision,
        'output_dir': args.model_dir
    }
    
    # Initialize model
    model = TelcoChurnDNN(config)
    
    # Build model
    model.build_model(input_dim=args.num_features)
    
    # Print model summary
    model.model.summary()
    
    # Find training files
    train_file = os.path.join(args.train, 'train.csv')
    val_file = os.path.join(args.validation, 'validation.csv')
    test_file = os.path.join(args.test, 'test.csv') if os.path.exists(os.path.join(args.test, 'test.csv')) else None
    
    # Train model
    logger.info("Starting distributed training with mixed precision...")
    history = model.train(
        train_path=train_file,
        val_path=val_file,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    # Evaluate on test set if available
    if test_file:
        logger.info("Evaluating on test set...")
        test_metrics = model.evaluate(test_file, args.batch_size)
    
    # Save model
    logger.info("Saving model artifacts...")
    model.save_model(args.model_dir)
    
    # Log final metrics
    final_metrics = {
        'final_train_loss': float(history.history['loss'][-1]),
        'final_val_loss': float(history.history['val_loss'][-1]),
        'final_val_auc': float(history.history['val_auc'][-1]),
        'best_val_auc': float(max(history.history['val_auc']))
    }
    
    logger.info(f"Training complete. Final metrics: {final_metrics}")
    
    # Write metrics for SageMaker
    with open(os.path.join(args.model_dir, 'metrics.json'), 'w') as f:
        json.dump(final_metrics, f)


if __name__ == '__main__':
    main()