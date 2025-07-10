#!/usr/bin/env python3
"""
Deploy endpoint only using existing trained model
"""

import boto3
import sagemaker
from sagemaker.tensorflow import TensorFlowModel
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def deploy_endpoint():
    """Deploy endpoint using the trained model"""
    
    session = sagemaker.Session()
    role_arn = "arn:aws:iam::108782097322:role/SageMakerExecutionRole"
    model_data = "s3://sagemaker-ap-southeast-3-108782097322/telco-churn/models/telco-churn-2025-07-10-12-15-41/output/model.tar.gz"
    
    timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    endpoint_name = f"telco-churn-endpoint-{timestamp}"
    
    logger.info(f"Creating endpoint: {endpoint_name}")
    logger.info(f"Using model: {model_data}")
    
    # Create TensorFlow model
    tf_model = TensorFlowModel(
        model_data=model_data,
        role=role_arn,
        framework_version='2.12',
        source_dir='../src',
        entry_point='inference.py'
    )
    
    # Deploy to endpoint
    predictor = tf_model.deploy(
        initial_instance_count=1,
        instance_type='ml.m5.large',
        endpoint_name=endpoint_name
    )
    
    logger.info(f"Endpoint deployed successfully!")
    logger.info(f"Endpoint name: {endpoint_name}")
    
    # Test the endpoint
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
        "Total Charges": 29.85,
        "CLTV": 3394
    }
    
    logger.info("Testing endpoint...")
    predictor.serializer = sagemaker.serializers.JSONSerializer()
    predictor.deserializer = sagemaker.deserializers.JSONDeserializer()
    
    result = predictor.predict(test_data)
    logger.info(f"Test result: {result}")
    
    return endpoint_name

if __name__ == '__main__':
    endpoint_name = deploy_endpoint()
    print(f"Endpoint deployed: {endpoint_name}")