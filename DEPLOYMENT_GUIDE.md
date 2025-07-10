# 🚀 SageMaker Deployment Guide: Telco Customer Churn Prediction

This guide shows how to deploy the **Naive Bayes vs TensorFlow DNN** churn prediction models to AWS SageMaker.

## 📋 Prerequisites

### AWS Setup
1. **AWS Account** with SageMaker permissions
2. **IAM Role** for SageMaker with these permissions:
   - `AmazonSageMakerFullAccess`
   - `AmazonS3FullAccess`
   - `IAMPassRoleAccess`

### Local Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Configure AWS credentials
aws configure
```

## 🏗️ Architecture Overview

```
Raw Data (Excel) → S3 → SageMaker Training Job → Model Artifacts → SageMaker Endpoint → Real-time Inference
```

### What Gets Deployed:
- **Training Script**: Trains both Naive Bayes and TensorFlow DNN
- **Inference Handler**: Serves the best performing model
- **Endpoint**: Real-time API for churn predictions

## 🚀 Quick Deployment

### Option 1: Automated Full Pipeline
```bash
cd deployment
python deploy.py --data-path ../data --keep-endpoint
```

### Option 2: Step-by-Step Deployment
```bash
# 1. Deploy with training
python deploy.py --data-path ../data --epochs 25 --batch-size 32

# 2. Test the endpoint  
python test_endpoint.py --endpoint-name <endpoint-name>

# 3. Cleanup when done
python deploy.py --cleanup --endpoint-name <endpoint-name>
```

## 📊 Expected Results

### Training Output:
```
Naive Bayes - Val AUC: 0.8234, Test AUC: 0.8156
TensorFlow DNN - Val AUC: 0.9012, Test AUC: 0.8987
Best model: TensorFlow DNN
```

### Inference Example:
```json
{
  "churn_probability": 0.8543,
  "churn_prediction": 1,
  "churn_label": "Yes",
  "model_used": "TensorFlow DNN",
  "confidence": "High"
}
```

## ⚙️ Configuration Options

### Training Hyperparameters:
```bash
python deploy.py \
  --epochs 30 \
  --batch-size 64 \
  --learning-rate 0.001 \
  --dropout-rate 0.3
```

### Instance Types:
- **Training**: `ml.m5.xlarge` (CPU, cost-effective)
- **Inference**: `ml.t2.medium` (low traffic) or `ml.m5.large` (high traffic)

## 🧪 Testing Your Deployment

### 1. Single Prediction Test
```python
test_customer = {
    "Gender": "Female",
    "Senior Citizen": 0,
    "Partner": "Yes",
    "Dependents": "No", 
    "Tenure Months": 1,
    "Monthly Charges": 85.0,
    "Contract": "Month-to-month",
    "Payment Method": "Electronic check"
    # ... other features
}
```

### 2. Comprehensive Testing
```bash
# Run all tests
python test_endpoint.py --endpoint-name <endpoint-name> --test-type comprehensive

# Performance test
python test_endpoint.py --endpoint-name <endpoint-name> --test-type performance --num-requests 50
```

## 💰 Cost Optimization

### Training Costs:
- **ml.m5.xlarge**: ~$0.192/hour
- **Typical training time**: 10-15 minutes
- **Estimated cost per training job**: ~$0.05

### Inference Costs:
- **ml.t2.medium**: ~$0.0464/hour (~$33/month)
- **ml.m5.large**: ~$0.096/hour (~$69/month)

### Cost-Saving Tips:
1. **Delete endpoints** when not in use
2. **Use smaller instances** for testing
3. **Enable auto-scaling** for production

## 🔧 Troubleshooting

### Common Issues:

#### 1. Permission Errors
```bash
# Check IAM role permissions
aws iam get-role --role-name SageMakerExecutionRole
```

#### 2. Training Job Failures
```bash
# Check CloudWatch logs
aws logs describe-log-groups --log-group-name-prefix /aws/sagemaker/TrainingJobs
```

#### 3. Endpoint Errors
```bash
# Test endpoint manually
python test_endpoint.py --endpoint-name <name> --test-type single
```

### Debug Mode:
```bash
# Enable verbose logging
export SAGEMAKER_LOG_LEVEL=DEBUG
python deploy.py --data-path ../data
```

## 📁 Project Structure After Deployment

```
telco-churn-sagemaker/
├── deployment/
│   ├── deploy.py                    # Main deployment script
│   ├── test_endpoint.py            # Endpoint testing
│   └── deployment_info.json       # Deployment details
├── src/
│   ├── training/
│   │   └── train.py               # SageMaker training script
│   └── inference/
│       └── inference.py           # SageMaker inference handler
├── data/
│   └── Telco_customer_churn.xlsx  # Training data
└── models/                        # Local model artifacts (optional)
```

## 🔍 Monitoring and Maintenance

### 1. Monitor Endpoint Performance
```bash
# CloudWatch metrics
aws cloudwatch get-metric-statistics \
  --namespace AWS/SageMaker \
  --metric-name Invocations \
  --dimensions Name=EndpointName,Value=<endpoint-name>
```

### 2. Model Retraining
```bash
# Retrain with new data
python deploy.py --data-path ../new_data --epochs 20
```

### 3. A/B Testing
```bash
# Deploy second model version
python deploy.py --model-version v2 --traffic-split 10
```

## 🎯 Production Checklist

- [ ] **IAM roles** configured correctly
- [ ] **Data validation** pipeline in place
- [ ] **Endpoint monitoring** set up
- [ ] **Auto-scaling** configured
- [ ] **Backup strategy** for models
- [ ] **Cost alerts** enabled
- [ ] **Security groups** configured
- [ ] **Logging** enabled in CloudWatch

## 🔗 Additional Resources

- [AWS SageMaker Documentation](https://docs.aws.amazon.com/sagemaker/)
- [TensorFlow on SageMaker](https://sagemaker.readthedocs.io/en/stable/frameworks/tensorflow/index.html)
- [SageMaker Python SDK](https://sagemaker.readthedocs.io/)
- [Cost Optimization Guide](https://docs.aws.amazon.com/sagemaker/latest/dg/inference-cost-optimization.html)

## 🆘 Support

For issues with this deployment:
1. Check the troubleshooting section above
2. Review CloudWatch logs
3. Test with the provided test scripts
4. Ensure all prerequisites are met

---

**🎉 Congratulations!** You now have a production-ready ML system showcasing TensorFlow's capabilities on AWS SageMaker!