#!/usr/bin/env python3
"""
SageMaker Deployment Script for Telco Customer Churn Prediction
Handles training job submission and endpoint deployment
"""

import boto3
import sagemaker
from sagemaker.tensorflow import TensorFlow
from sagemaker.sklearn import SKLearn
import os
import json
import time
from datetime import datetime
import argparse
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChurnModelDeployer:
    """Handles SageMaker training and deployment for churn prediction models"""
    
    def __init__(self, role_arn=None, region='us-east-1'):
        """Initialize the deployer with AWS configurations"""
        self.region = region
        self.session = sagemaker.Session()
        self.role_arn = role_arn or sagemaker.get_execution_role()
        self.bucket = self.session.default_bucket()
        
        # Generate unique identifiers
        timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        self.job_name = f"telco-churn-{timestamp}"
        self.model_name = f"telco-churn-model-{timestamp}"
        self.endpoint_name = f"telco-churn-endpoint-{timestamp}"
        
        logger.info(f"Initialized deployer:")
        logger.info(f"  Region: {self.region}")
        logger.info(f"  Bucket: {self.bucket}")
        logger.info(f"  Role: {self.role_arn}")
        logger.info(f"  Job name: {self.job_name}")
    
    def upload_data(self, local_data_path):
        """Upload training data to S3"""
        logger.info("Uploading training data to S3...")
        
        s3_data_path = f"s3://{self.bucket}/telco-churn/data"
        
        # Upload the Excel file
        s3_client = boto3.client('s3')
        
        # Find the data file
        data_file = None
        for filename in os.listdir(local_data_path):
            if filename.endswith(('.xlsx', '.csv')):
                data_file = os.path.join(local_data_path, filename)
                break
        
        if not data_file:
            raise FileNotFoundError(f"No data file found in {local_data_path}")
        
        # Upload to S3
        key = f"telco-churn/data/{os.path.basename(data_file)}"
        s3_client.upload_file(data_file, self.bucket, key)
        
        logger.info(f"Data uploaded to: s3://{self.bucket}/{key}")
        return s3_data_path
    
    def upload_code(self, source_dir):
        """Upload training code to S3"""
        logger.info("Uploading training code...")
        
        # Create a tarball of the source code
        import tarfile
        import tempfile
        
        with tempfile.NamedTemporaryFile(suffix='.tar.gz', delete=False) as tmp:
            with tarfile.open(tmp.name, 'w:gz') as tar:
                # Add training script
                tar.add(os.path.join(source_dir, 'training', 'train.py'), 
                       arcname='train.py')
                # Add inference script
                tar.add(os.path.join(source_dir, 'inference', 'inference.py'), 
                       arcname='inference.py')
                # Add requirements if exists
                req_file = os.path.join(os.path.dirname(source_dir), 'requirements.txt')
                if os.path.exists(req_file):
                    tar.add(req_file, arcname='requirements.txt')
            
            # Upload to S3
            s3_client = boto3.client('s3')
            key = f"telco-churn/code/source.tar.gz"
            s3_client.upload_file(tmp.name, self.bucket, key)
            
            os.unlink(tmp.name)  # Clean up temp file
        
        logger.info(f"Code uploaded to: s3://{self.bucket}/{key}")
        return f"s3://{self.bucket}/{key}"
    
    def create_training_job(self, data_location, hyperparameters=None):
        """Create and submit a SageMaker training job"""
        logger.info("Creating SageMaker training job...")
        
        # Default hyperparameters
        if hyperparameters is None:
            hyperparameters = {
                'epochs': 20,
                'batch-size': 64,
                'learning-rate': 0.001,
                'dropout-rate': 0.3
            }
        
        # Create TensorFlow estimator
        estimator = TensorFlow(
            entry_point='train.py',
            source_dir='src',
            role=self.role_arn,
            instance_count=1,
            instance_type='ml.m5.xlarge',  # CPU instance for training
            framework_version='2.12',
            py_version='py310',
            script_mode=True,
            hyperparameters=hyperparameters,
            output_path=f"s3://{self.bucket}/telco-churn/models",
            code_location=f"s3://{self.bucket}/telco-churn/code",
            base_job_name='telco-churn-training',
            enable_sagemaker_metrics=True,
            metric_definitions=[
                {'Name': 'train:auc', 'Regex': 'Train AUC: ([0-9\\.]+)'},
                {'Name': 'validation:auc', 'Regex': 'Val AUC: ([0-9\\.]+)'},
                {'Name': 'test:auc', 'Regex': 'Test AUC: ([0-9\\.]+)'}
            ]
        )
        
        # Set up data channels
        train_input = sagemaker.inputs.TrainingInput(
            s3_data=data_location,
            content_type='application/x-excel'
        )
        
        # Start training
        logger.info(f"Starting training job: {self.job_name}")
        estimator.fit({
            'train': train_input,
            'validation': train_input,  # Same data for simplicity
            'test': train_input
        }, job_name=self.job_name)
        
        logger.info("Training job completed successfully!")
        return estimator
    
    def deploy_model(self, estimator, instance_type='ml.t2.medium'):
        """Deploy the trained model to a SageMaker endpoint"""
        logger.info(f"Deploying model to endpoint: {self.endpoint_name}")
        
        # Deploy to endpoint
        predictor = estimator.deploy(
            initial_instance_count=1,
            instance_type=instance_type,
            endpoint_name=self.endpoint_name,
            wait=True
        )
        
        logger.info(f"Model deployed successfully!")
        logger.info(f"Endpoint name: {self.endpoint_name}")
        logger.info(f"Endpoint URL: https://{self.region}.console.aws.amazon.com/sagemaker/home?region={self.region}#/endpoints/{self.endpoint_name}")
        
        return predictor
    
    def test_endpoint(self, predictor):
        """Test the deployed endpoint with sample data"""
        logger.info("Testing the deployed endpoint...")
        
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
            # Set content type for JSON
            predictor.serializer = sagemaker.serializers.JSONSerializer()
            predictor.deserializer = sagemaker.deserializers.JSONDeserializer()
            
            # Make prediction
            result = predictor.predict(test_data)
            
            logger.info("Test prediction successful!")
            logger.info(f"Result: {json.dumps(result, indent=2)}")
            
            return result
            
        except Exception as e:
            logger.error(f"Test prediction failed: {e}")
            raise
    
    def cleanup_endpoint(self, predictor=None):
        """Clean up the endpoint to avoid charges"""
        try:
            if predictor:
                logger.info(f"Deleting endpoint: {self.endpoint_name}")
                predictor.delete_endpoint()
                logger.info("Endpoint deleted successfully!")
            else:
                # Delete using boto3 if no predictor available
                sagemaker_client = boto3.client('sagemaker', region_name=self.region)
                sagemaker_client.delete_endpoint(EndpointName=self.endpoint_name)
                logger.info("Endpoint deleted successfully!")
        except Exception as e:
            logger.warning(f"Error deleting endpoint: {e}")
    
    def full_deployment_pipeline(self, data_path, test_endpoint=True, keep_endpoint=False, endpoint_instance_type='ml.m5.large'):
        """Run the complete deployment pipeline"""
        logger.info("Starting full deployment pipeline...")
        
        try:
            # 1. Upload data
            s3_data_path = self.upload_data(data_path)
            
            # 2. Create and run training job
            estimator = self.create_training_job(s3_data_path)
            
            # 3. Deploy model
            predictor = self.deploy_model(estimator, endpoint_instance_type)
            
            # 4. Test endpoint
            if test_endpoint:
                self.test_endpoint(predictor)
            
            # 5. Provide deployment info
            deployment_info = {
                'endpoint_name': self.endpoint_name,
                'model_name': self.model_name,
                'training_job_name': self.job_name,
                'region': self.region,
                'bucket': self.bucket,
                'deployment_time': datetime.now().isoformat(),
                'endpoint_url': f"https://{self.region}.console.aws.amazon.com/sagemaker/home?region={self.region}#/endpoints/{self.endpoint_name}"
            }
            
            # Save deployment info
            with open('deployment_info.json', 'w') as f:
                json.dump(deployment_info, f, indent=2)
            
            logger.info("Deployment pipeline completed successfully!")
            logger.info(f"Deployment info saved to: deployment_info.json")
            
            # Cleanup if requested
            if not keep_endpoint:
                logger.info("Cleaning up endpoint (set --keep-endpoint to retain)...")
                time.sleep(10)  # Give time to see results
                self.cleanup_endpoint(predictor)
            else:
                logger.info(f"Endpoint kept running: {self.endpoint_name}")
                logger.info("Remember to delete it manually to avoid charges!")
            
            return deployment_info
            
        except Exception as e:
            logger.error(f"Deployment pipeline failed: {e}")
            # Try to cleanup on failure
            try:
                self.cleanup_endpoint()
            except:
                pass
            raise

def main():
    parser = argparse.ArgumentParser(description='Deploy Telco Churn Model to SageMaker')
    
    parser.add_argument('--data-path', type=str, default='data',
                       help='Path to directory containing training data')
    parser.add_argument('--role-arn', type=str, default=None,
                       help='SageMaker execution role ARN')
    parser.add_argument('--region', type=str, default='us-east-1',
                       help='AWS region')
    parser.add_argument('--instance-type', type=str, default='ml.t2.medium',
                       help='Instance type for endpoint')
    parser.add_argument('--keep-endpoint', action='store_true',
                       help='Keep endpoint running after deployment')
    parser.add_argument('--test-only', action='store_true',
                       help='Only test existing endpoint')
    parser.add_argument('--endpoint-name', type=str, 
                       help='Existing endpoint name for testing')
    
    # Hyperparameters
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--dropout-rate', type=float, default=0.3)
    
    args = parser.parse_args()
    
    try:
        if args.test_only:
            # Test existing endpoint
            if not args.endpoint_name:
                raise ValueError("--endpoint-name required for --test-only mode")
            
            logger.info(f"Testing existing endpoint: {args.endpoint_name}")
            
            # Create a predictor for existing endpoint
            from sagemaker.predictor import Predictor
            predictor = Predictor(
                endpoint_name=args.endpoint_name,
                sagemaker_session=sagemaker.Session()
            )
            
            deployer = ChurnModelDeployer(args.role_arn, args.region)
            deployer.endpoint_name = args.endpoint_name
            deployer.test_endpoint(predictor)
            
        else:
            # Full deployment pipeline
            hyperparameters = {
                'epochs': args.epochs,
                'batch-size': args.batch_size,
                'learning-rate': args.learning_rate,
                'dropout-rate': args.dropout_rate
            }
            
            deployer = ChurnModelDeployer(args.role_arn, args.region)
            deployer.full_deployment_pipeline(
                data_path=args.data_path,
                test_endpoint=True,
                keep_endpoint=args.keep_endpoint,
                endpoint_instance_type=args.instance_type
            )
        
        logger.info("Script completed successfully!")
        
    except Exception as e:
        logger.error(f"Script failed: {e}")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())